"""Differentiable Weightless Controllers (DWCs) as piecewise-constant grid policies.

A DWC observes each state dimension through a thermometer code. Its thresholds
partition the state space into axis-aligned policy cells, and the hard LUT
network maps the thermometer code of a cell to integer popcounts, the *native
values*. An affine decoder turns native values into actions. A repair
overrides the native value of individual cells and leaves the network intact.

Grid convention. Along dimension ``d`` with finite thresholds
``t_0 < ... < t_{b-1}``, cell ``i`` is ``(t_{i-1}, t_i]`` with ``t_{-1} = -inf``
and ``t_b = +inf``: a finite threshold belongs to the cell on its left.
Flattened cell ids are first-dimension-fast. Thermometer bit ``j`` of
dimension ``d`` is ``clamp(x_d, lower_d, upper_d) > t_j``.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch

FORMAT = "reacheco-dwc"
VERSION = 1
_INFERENCE_BATCH = 2048


class DWCPolicy:
    """A hard DWC evaluated lazily per policy cell, with per-cell overrides."""

    def __init__(self, artifact, *, device="cpu"):
        _validate_artifact(artifact)
        grid, network = artifact["grid"], artifact["network"]
        self.device = torch.device(device)
        self.metadata = copy.deepcopy(artifact.get("metadata", {}))

        self.lower = grid["lower"].to(self.device, torch.float32)
        self.upper = grid["upper"].to(self.device, torch.float32)
        finite = [row.to(self.device, torch.float32) for row in grid["thresholds"]]
        # Each threshold row ends in +inf so that searchsorted yields cell indices.
        self.thresholds = tuple(torch.cat((row, row.new_tensor([math.inf]))) for row in finite)
        self.sizes = torch.tensor([len(row) for row in self.thresholds], dtype=torch.long, device=self.device)
        self.strides = torch.cumprod(torch.cat((self.sizes.new_ones(1), self.sizes[:-1])), 0)
        self.num_cells = int(self.sizes.prod())
        # Clamping the observation to [lower, upper] fixes the bits of thresholds outside it.
        # for our experiments this is not relevant.
        self._bits_on = [row < lo for row, lo in zip(finite, self.lower)]
        self._bits_off = [row >= hi for row, hi in zip(finite, self.upper)]

        self.mappings = [m.to(self.device, torch.long) for m in network["mappings"]]
        self.luts = [lut.to(self.device, torch.bool) for lut in network["luts"]]
        # exp on the CPU: the decoder and the repair bound must not depend on the device.
        self.log_alpha = network["log_alpha"].to(torch.float32)
        self.alpha = self.log_alpha.exp().to(self.device)
        self.beta = network["beta"].to(self.device, torch.float32)
        self.output_tanh = bool(network["output_tanh"])
        self.action_clip = tuple(float(v) for v in network["action_clip"])
        self.action_dim = len(self.beta)
        self.output_count = len(self.luts[-1]) // self.action_dim
        self.min_value, self.max_value = self._repair_range()

        self._network_cache = {}
        self.overrides = {}
        self._override_table = None
        overrides = artifact.get("overrides")
        if overrides is not None and len(overrides["cell_ids"]):
            self.set_cell_values(overrides["cell_ids"], overrides["values"])

    # ------------------------------------------------------------------ grid

    def cell_ids(self, points):
        """Flattened ids of the cells containing ``points`` of shape ``(N, D)``."""
        ids = torch.zeros(len(points), dtype=torch.long, device=points.device)
        for d, row in enumerate(self.thresholds):
            ids += torch.searchsorted(row.to(points), points[:, d].contiguous()) * int(self.strides[d])
        return ids

    def box_cell_ids(self, boxes):
        """Cell ids of policy-aligned boxes ``(N, 2, D)``, read at their upper corner.

        Aligned boxes are closed and their lower face may lie on the threshold
        that belongs to the previous cell; the upper corner always lies in the
        box's own cell (``+inf`` maps to the outermost cell).
        """
        return self.cell_ids(boxes[:, 1])

    def _coordinates(self, ids):
        return (ids[:, None] // self.strides) % self.sizes

    def _check_ids(self, cell_ids):
        ids = torch.as_tensor(cell_ids, dtype=torch.long, device=self.device).reshape(-1)
        if bool(((ids < 0) | (ids >= self.num_cells)).any()):
            raise IndexError("policy cell id outside the grid")
        return ids

    # -------------------------------------------------------------- values

    def network_values(self, cell_ids):
        """Native values of the unrepaired network, ``(N, action_dim)`` float32."""
        ids = self._check_ids(cell_ids)
        unique, inverse = torch.unique(ids, return_inverse=True)
        keys = unique.tolist()
        missing = [key for key in keys if key not in self._network_cache]
        if missing:
            inferred = self._infer(torch.tensor(missing, device=self.device))
            self._network_cache.update(zip(missing, map(tuple, inferred.tolist())))
        rows = [self._network_cache[key] for key in keys]
        return self.lower.new_tensor(rows).reshape(-1, self.action_dim)[inverse]

    def cell_values(self, cell_ids):
        """Native values of the repaired policy, ``(N, action_dim)`` float32."""
        ids = self._check_ids(cell_ids)
        values = self.network_values(ids)
        if self.overrides:
            keys, rows = self._override_tensors()
            position = torch.searchsorted(keys, ids).clamp_max(len(keys) - 1)
            hit = keys[position] == ids
            values[hit] = rows[position[hit]]
        return values

    def _override_tensors(self):
        if self._override_table is None:
            keys = sorted(self.overrides)
            self._override_table = (
                torch.tensor(keys, dtype=torch.long, device=self.device),
                self.lower.new_tensor([self.overrides[key] for key in keys]).reshape(-1, self.action_dim),
            )
        return self._override_table

    def box_values(self, boxes):
        """Native values selected in each policy-aligned box."""
        return self.cell_values(self.box_cell_ids(boxes))

    def set_cell_values(self, cell_ids, values):
        """Override native values; a value equal to the network's removes the override."""
        ids = self._check_ids(cell_ids)
        values = self._check_values(values).reshape(-1, self.action_dim)
        if len(ids) != len(values):
            raise ValueError("cell ids and values must have equal length")
        assigned = {}
        for key, value in zip(ids.tolist(), map(tuple, values.long().tolist())):
            if assigned.setdefault(key, value) != value:
                raise ValueError(f"cell {key} received two different values")
        network = dict(zip(assigned, map(tuple, self.network_values(list(assigned)).tolist())))
        for key, value in assigned.items():
            if value == network[key]:
                self.overrides.pop(key, None)
            else:
                self.overrides[key] = value
        self._override_table = None

    def copy(self):
        """An independent copy of the overrides; the network and its value cache are shared."""
        result = copy.copy(self)
        result.overrides = dict(self.overrides)
        result._override_table = None
        return result

    def changed_cells(self, other):
        """Cells on which this policy and ``other`` select different native values."""
        keys = sorted(set(self.overrides) | set(other.overrides))
        if not keys:
            return frozenset()
        differs = (self.cell_values(keys) != other.cell_values(keys)).any(-1)
        return frozenset(key for key, flag in zip(keys, differs.tolist()) if flag)

    # ------------------------------------------------------------- actions

    def decode(self, values):
        """Map native values ``(N, action_dim)`` to actions."""
        values = self._check_values(values)
        actions = self.alpha * (values / float(self.output_count) - 0.5) + self.beta
        if self.output_tanh:
            actions = torch.tanh(actions)
        return actions.clamp(*self.action_clip)

    def repair_candidates(self, count):
        """``count`` evenly spaced native values per component, ``(K, action_dim)``."""
        if int(count) < 2:
            raise ValueError("a repair alphabet needs at least two values")
        axis = torch.linspace(self.min_value, self.max_value, int(count), device=self.device).round().unique()
        if self.action_dim == 1:
            return axis[:, None]
        return torch.cartesian_prod(*[axis] * self.action_dim)

    def _repair_range(self):
        """Smallest and largest native value a repair may assign.

        Native popcounts span ``0 .. output_count``. For a scalar affine decoder
        the range is extended in both directions until both action clips are
        representable (for ACC, full acceleration and full braking). Values
        outside ``0 .. output_count`` are overrides before the decoder, never
        popcounts of the network. Tanh and multi-component decoders keep the
        native range.
        """
        native = self.output_count
        if self.output_tanh or self.action_dim != 1:
            return 0, native
        low, high = self.action_clip

        def required(action):
            value = native * (0.5 + (action - float(self.beta)) / float(self.alpha))
            if not math.isfinite(value) or abs(value) > 2**24:
                raise ValueError("repair range exceeds the exact float32 integers")
            return value

        minimum = min(0, math.floor(required(low)))
        while self._decode_unchecked(minimum) > low:
            minimum -= 1
        maximum = max(native, math.ceil(required(high)))
        while self._decode_unchecked(maximum) < high:
            maximum += 1
        return minimum, maximum

    def _decode_unchecked(self, value):
        actions = self.alpha * (self.lower.new_tensor([[float(value)]]) / float(self.output_count) - 0.5) + self.beta
        return float(actions.clamp(*self.action_clip))

    def _check_values(self, values):
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if (
            not bool(torch.isfinite(values).all())
            or not torch.equal(values, values.round())
            or bool((values < self.min_value).any())
            or bool((values > self.max_value).any())
        ):
            raise ValueError(f"native values must be integers in [{self.min_value}, {self.max_value}]")
        return values

    # ----------------------------------------------------------- inference

    def _infer(self, ids):
        """Run the LUT network on the thermometer codes of the given cells.
        Reimplementation of Bacellar 2024.
        """
        outputs = []
        for batch in ids.split(_INFERENCE_BATCH):
            coordinates = self._coordinates(batch)
            codes = []
            for d, (on, off) in enumerate(zip(self._bits_on, self._bits_off)):
                bits = torch.arange(len(on), device=self.device)[None] < coordinates[:, d, None]
                codes.append((bits & ~off) | on)
            values = torch.cat(codes, 1)
            for mapping, lut in zip(self.mappings, self.luts):
                powers = 2 ** torch.arange(mapping.shape[1], device=self.device)
                addresses = (values[:, mapping].long() * powers).sum(-1)
                rows = torch.arange(len(lut), device=self.device) * lut.shape[1]
                values = lut.reshape(-1)[rows + addresses]
            outputs.append(values.reshape(len(batch), self.action_dim, self.output_count).sum(-1))
        return torch.cat(outputs).to(torch.float32)

    # ---------------------------------------------------------- artifacts

    def to_artifact(self):
        ids = sorted(self.overrides)
        return {
            "format": FORMAT,
            "version": VERSION,
            "grid": {
                "thresholds": [row[:-1].cpu() for row in self.thresholds],
                "lower": self.lower.cpu(),
                "upper": self.upper.cpu(),
            },
            "network": {
                "mappings": [m.cpu() for m in self.mappings],
                "luts": [lut.cpu() for lut in self.luts],
                "log_alpha": self.log_alpha.clone(),
                "beta": self.beta.cpu(),
                "output_tanh": self.output_tanh,
                "action_clip": list(self.action_clip),
            },
            "overrides": {
                "cell_ids": torch.tensor(ids, dtype=torch.long),
                "values": torch.tensor([self.overrides[key] for key in ids], dtype=torch.long).reshape(-1, self.action_dim),
            },
            "metadata": copy.deepcopy(self.metadata),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_artifact(), path)
        return path

    @classmethod
    def load(cls, path, *, device="cpu"):
        artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(artifact, device=device)


def _validate_artifact(artifact):
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} artifact")
    if artifact.get("version") != VERSION:
        raise ValueError(f"unsupported {FORMAT} version {artifact.get('version')!r}")
    grid, network = artifact["grid"], artifact["network"]

    rows, lower, upper = grid["thresholds"], grid["lower"], grid["upper"]
    if lower.ndim != 1 or lower.shape != upper.shape or len(rows) != len(lower):
        raise ValueError("grid needs one threshold row and one bound pair per dimension")
    if not bool((lower < upper).all()):
        raise ValueError("grid bounds must be increasing")
    for row in rows:
        if row.ndim != 1 or not bool(torch.isfinite(row).all()) or not bool((row[1:] > row[:-1]).all()):
            raise ValueError("thresholds must be finite and strictly increasing")

    width = sum(len(row) for row in rows)
    mappings, luts = network["mappings"], network["luts"]
    if not mappings or len(mappings) != len(luts):
        raise ValueError("the network needs one LUT table per mapping")
    for layer, (mapping, lut) in enumerate(zip(mappings, luts)):
        if (
            mapping.ndim != 2 or lut.ndim != 2 or len(mapping) != len(lut)
            or lut.shape[1] != 1 << mapping.shape[1]
            or bool((mapping < 0).any()) or bool((mapping >= width).any())
        ):
            raise ValueError(f"invalid mapping or LUT in layer {layer}")
        width = len(mapping)

    beta, log_alpha = network["beta"], network["log_alpha"]
    if beta.ndim != 1 or beta.shape != log_alpha.shape or len(beta) == 0 or width % len(beta):
        raise ValueError("invalid regression head")
    low, high = network["action_clip"]
    if not low < high:
        raise ValueError("action_clip must be increasing")
