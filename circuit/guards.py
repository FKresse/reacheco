"""The cluster guard decoder: which repair rectangle, if any, contains the input.

A repaired controller differs from its base network on a set of grid cells. Each
maximal rectangle of that set is a *cluster*, described by an inclusive interval
of cell indices per state dimension. The decoder answers, for one observation,
which cluster contains it -- as a one-hot signal -- and it does so from the
thermometer bits the controller already computes, so it needs no arithmetic and
no second copy of the input encoding.

The construction rests on one identity. With thermometer thresholds
``t[0] < ... < t[B-1]`` the encoded bit ``j`` is the predicate ``x > t[j]``, and
the cell index of ``x`` is the number of thresholds it exceeds. So

    cell(x) >= lo   <=>   x > t[lo - 1]   <=>   bit[lo - 1]
    cell(x) <= hi   <=>   x <= t[hi]      <=>   NOT bit[hi]

and a one-dimensional interval guard is a single two-input LUT over two
thermometer bits. Either half drops out when it is vacuous: ``lo == 0`` has no
lower bound, ``hi == B`` is the unbounded top cell, and a threshold outside the
grid makes its bit constant. The per-dimension guards are then combined by a
single AND per cluster (``build_and_layer``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .lut import AND_2, AND_NOT, NOT_1, TRUE, LutLayer, and_table


@dataclass(frozen=True)
class GridCode:
    """How a thermometer-coded controller turns a state into decoder inputs.

    ``bits[d]`` thresholds per observed dimension ``d`` induce ``bits[d] + 1``
    cells. ``always_true`` and ``always_false`` mark thresholds that fall outside
    the grid the verifier reasons over, whose bits are therefore constant and
    cannot be used as guard inputs.
    """

    bits: Sequence[int]
    always_true: Sequence[torch.Tensor]
    always_false: Sequence[torch.Tensor]

    @property
    def dims(self) -> int:
        return len(self.bits)

    @property
    def cells(self):
        return tuple(int(b) + 1 for b in self.bits)

    @property
    def input_size(self) -> int:
        return int(sum(int(b) for b in self.bits))

    @property
    def offsets(self):
        """Where each dimension's bits start in the concatenated code."""
        return torch.tensor([0] + [int(b) for b in self.bits], dtype=torch.long).cumsum(0)

    def thermometer(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode cell coordinates ``(N, dims)`` the way the controller does."""
        codes = []
        for dim in range(self.dims):
            bit = torch.arange(int(self.bits[dim]), device=coordinates.device)
            code = bit[None] < coordinates[:, dim, None]
            code = code & ~self.always_false[dim].to(coordinates.device)[None]
            code = code | self.always_true[dim].to(coordinates.device)[None]
            codes.append(code)
        return torch.cat(codes, dim=1)


def _guard_bits(code: GridCode, dim: int, lower: int, upper: int):
    """The two thermometer bits an interval guard tests, or None where vacuous."""
    if lower > upper:
        raise ValueError(f"empty guard [{lower}, {upper}] on dimension {dim}")
    if upper >= code.cells[dim]:
        raise ValueError(f"guard [{lower}, {upper}] leaves the {code.cells[dim]}-cell grid")

    low_bit: Optional[int] = None if lower == 0 else lower - 1
    if low_bit is not None:
        if bool(code.always_false[dim][low_bit]):
            raise ValueError(
                f"dimension {dim}: cells above {lower - 1} are unreachable inside the grid, "
                f"so the guard [{lower}, {upper}] covers nothing")
        if bool(code.always_true[dim][low_bit]):
            low_bit = None          # the lower bound holds everywhere in the grid

    high_bit: Optional[int] = None if upper == int(code.bits[dim]) else upper
    if high_bit is not None:
        if bool(code.always_true[dim][high_bit]):
            raise ValueError(
                f"dimension {dim}: every state exceeds threshold {upper} inside the grid, "
                f"so the guard [{lower}, {upper}] covers nothing")
        if bool(code.always_false[dim][high_bit]):
            high_bit = None         # the upper bound holds everywhere in the grid
    return low_bit, high_bit


def build_interval_layer(code: GridCode, guards: Sequence[torch.Tensor]):
    """One two-input LUT per distinct interval guard, over the thermometer bits.

    ``guards[d]`` holds the distinct ``[lower, upper]`` cell intervals appearing
    on dimension ``d``. The returned layer emits them in dimension order, which
    ``offsets`` indexes into.
    """
    widths = [int(len(g)) for g in guards]
    offsets = torch.tensor([0] + widths, dtype=torch.long).cumsum(0)
    bit_offsets = code.offsets
    total = int(offsets[-1])

    mapping = torch.zeros((total, 2), dtype=torch.long)
    luts = torch.empty((total, 4), dtype=torch.int8)
    for dim in range(code.dims):
        base = int(bit_offsets[dim])
        for index, (lower, upper) in enumerate(guards[dim].tolist()):
            out = int(offsets[dim]) + index
            low_bit, high_bit = _guard_bits(code, dim, int(lower), int(upper))
            if low_bit is not None and high_bit is not None:
                mapping[out] = torch.tensor([base + low_bit, base + high_bit])
                luts[out] = torch.tensor(AND_NOT, dtype=torch.int8)
            elif low_bit is not None:                      # x > t[low_bit]
                mapping[out] = base + low_bit
                luts[out] = torch.tensor(AND_2, dtype=torch.int8)
            elif high_bit is not None:                     # x <= t[high_bit]
                mapping[out] = base + high_bit
                luts[out] = torch.tensor(NOT_1, dtype=torch.int8)
            else:                                          # the whole dimension
                mapping[out] = base
                luts[out] = torch.tensor(TRUE, dtype=torch.int8)
    return LutLayer(code.input_size, mapping, luts), offsets


def build_and_layer(guard_index: torch.Tensor, input_size: int):
    """AND the per-dimension guards of every rectangle into one signal.

    ``guard_index[c, d]`` is the interval-layer output holding dimension ``d``
    of rectangle ``c``. With at most six dimensions one LUT6 ANDs all of them;
    a single dimension needs no gate. Identical guard tuples share their gate.
    Returns the layers and each rectangle's output index.
    """
    dims = guard_index.shape[1]
    if dims > 6:
        raise ValueError("more than six dimensions need a deeper AND tree")
    if dims == 1:
        return [], guard_index[:, 0].clone()
    unique, inverse = torch.unique(guard_index, dim=0, return_inverse=True)
    return [LutLayer(input_size, unique, and_table(dims).expand(len(unique), -1).clone())], inverse


def build_guard_decoder(code: GridCode, lower: torch.Tensor, upper: torch.Tensor):
    """Full decoder for clusters ``[lower[c], upper[c]]`` in cell-index space.

    Returns the layer chain, the index of each cluster's one-hot output in the
    final layer, and the number of LUTs.
    """
    if lower.shape != upper.shape:
        raise ValueError("cluster corners must have the same shape")
    if lower.shape[1] != code.dims:
        raise ValueError(f"clusters span {lower.shape[1]} dimensions, the code has {code.dims}")
    if bool((lower > upper).any()):
        raise ValueError("a cluster has an inverted corner")

    guards, inverse = [], []
    for dim in range(code.dims):
        unique, index = torch.unique(torch.stack((lower[:, dim], upper[:, dim]), dim=1),
                                     dim=0, return_inverse=True)
        guards.append(unique)
        inverse.append(index)

    interval_layer, offsets = build_interval_layer(code, guards)
    guard_index = torch.stack([offsets[dim] + inverse[dim] for dim in range(code.dims)], dim=1)
    and_layers, one_hot_index = build_and_layer(guard_index, interval_layer.output_size)
    layers = [interval_layer, *and_layers]
    return layers, one_hot_index, sum(layer.output_size for layer in layers)
