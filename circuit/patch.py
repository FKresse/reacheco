"""Turn a repaired DWC controller into a BRAM patch circuit and measure it.

A repaired controller keeps the network untouched and stores the repair as
per-cell overrides of the native values (``DWCPolicy.overrides``), so it
describes the patch on its own: the delta of a cell is its override minus the
network's value there. Each repaired cell becomes one cluster.

The compiled circuit is verified before it is reported: grid cells are pushed
through the decoder, and on every one the circuit's delta must equal the
repaired policy's (the override minus the network value, zero elsewhere).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import cost
from .guards import GridCode, build_guard_decoder
from .lut import LutLayer, evaluate


@dataclass(frozen=True)
class ClusterPatch:
    """Repair rectangles in cell-index space with one signed delta each.

    ``lower`` and ``upper`` are inclusive corners ``(clusters, dims)``,
    ``deltas`` the native integer change per action dimension.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    deltas: torch.Tensor

    def __post_init__(self):
        if not (len(self.lower) == len(self.upper) == len(self.deltas)):
            raise ValueError("cluster corners and deltas must have equal length")

    def __len__(self):
        return int(self.lower.shape[0])

    @property
    def cells_covered(self) -> int:
        return int((self.upper - self.lower + 1).prod(dim=1).sum())


def grid_code(policy) -> GridCode:
    """How the controller's thermometer encodes the state.

    A threshold outside the grid bounds yields a constant bit (the policy clamps
    the state), which cannot fence a rectangle.
    """
    bits, always_true, always_false = [], [], []
    for row, lower, upper in zip(policy.thresholds, policy.lower.cpu(), policy.upper.cpu()):
        thresholds = row[:-1].cpu()                 # the last entry is +inf
        bits.append(int(thresholds.numel()))
        always_true.append(thresholds < lower)
        always_false.append(thresholds >= upper)
    return GridCode(tuple(bits), tuple(always_true), tuple(always_false))


def cell_coordinates(policy, cell_ids) -> torch.Tensor:
    """Grid coordinates ``(N, dims)`` of flattened cell ids."""
    ids = torch.as_tensor(cell_ids, dtype=torch.long).reshape(-1)
    return (ids[:, None] // policy.strides.to(ids.device)) % policy.sizes.to(ids.device)


def cell_clusters(policy) -> ClusterPatch:
    """One cluster per repaired cell."""
    keys = sorted(policy.overrides)
    if not keys:
        empty = torch.zeros((0, len(policy.thresholds)), dtype=torch.long)
        return ClusterPatch(empty, empty.clone(), torch.zeros((0, policy.action_dim), dtype=torch.long))
    coordinates = cell_coordinates(policy, keys)
    deltas = (policy.cell_values(keys) - policy.network_values(keys)).cpu().long()
    return ClusterPatch(coordinates, coordinates.clone(), deltas)


def _check_cells(patch, layers, one_hot_index, words, ids, code, policy, keys, deltas):
    """Decode one batch of cells and compare the circuit's delta with the policy's (``deltas`` at ``keys``)."""
    selected = evaluate(layers, code.thermometer(cell_coordinates(policy, ids)))
    one_hot = selected[:, one_hot_index]
    hits = one_hot.sum(dim=1)
    if bool((hits > 1).any()):
        row = int(torch.nonzero(hits > 1)[0])
        raise AssertionError(f"repair rectangles overlap: cell {int(ids[row])} matches clusters "
                             f"{torch.nonzero(one_hot[row]).reshape(-1).tolist()}")
    address = (one_hot.long() * torch.arange(1, len(patch) + 1, device=ids.device)[None]).sum(dim=1)
    circuit = words[:, address].transpose(0, 1)
    expected = torch.zeros_like(circuit)
    position = torch.searchsorted(keys, ids).clamp(max=len(keys) - 1)
    repaired = keys[position] == ids
    expected[repaired] = deltas[position[repaired]]
    if not torch.equal(circuit, expected):
        row = int(torch.nonzero((circuit != expected).any(dim=1))[0])
        raise AssertionError(f"the circuit disagrees with the repaired policy at cell {int(ids[row])}: "
                             f"delta {circuit[row].tolist()} vs {expected[row].tolist()}")
    return int((hits > 0).sum())


def verify_patch(policy, patch, layers, one_hot_index, words, *, batch_size=8192, max_cells=4_000_000,
                 seed=0) -> dict:
    """Decode grid cells and compare the circuit against the repaired policy.

    Checks that the guards are disjoint, that address zero means "no repair",
    and that every stored delta is the repaired one. Every cell is checked when
    the grid has at most ``max_cells``; otherwise every repaired cell plus a
    uniform sample of the rest, as ``scope`` says. Runs on the GPU if there is one.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code = grid_code(policy)
    total = policy.num_cells
    keys = torch.tensor(sorted(policy.overrides), dtype=torch.long)
    deltas = (policy.cell_values(keys) - policy.network_values(keys)).long().to(device) if len(keys) else None
    if total <= max_cells:
        ids_to_check, scope = torch.arange(total), "exhaustive"
    else:
        generator = torch.Generator().manual_seed(int(seed))
        sample = torch.randint(0, total, (int(max_cells),), generator=generator)
        ids_to_check = torch.unique(torch.cat((keys, sample)))
        scope = "every repaired cell plus a uniform sample"
    layers = [LutLayer(layer.input_size, layer.mapping.to(device), layer.luts.to(device)) for layer in layers]
    matched = sum(_check_cells(patch, layers, one_hot_index.to(device), words.to(device), ids, code, policy,
                               keys.to(device), deltas)
                  for ids in ids_to_check.to(device).split(batch_size))
    return {"scope": scope, "grid_cells": total, "cells_checked": int(len(ids_to_check)),
            "cells_matching_a_cluster": matched, "cells_repaired": patch.cells_covered,
            "exact": scope == "exhaustive"}


def compile_patch(policy, patch=None, *, verify=True, label="patch") -> dict:
    """Build the decoder and memory for ``patch`` (default: one cluster per
    repaired cell), verify them, and report their cost in LUT6 and memory bits."""
    patch = cell_clusters(policy) if patch is None else patch
    summary = {"label": label, "clusters": len(patch), "cells_repaired": patch.cells_covered,
               "guard_lut6": 0, "encoder_lut6": 0, "adder_lut6": 0, "total_lut6": 0, "memory_bits": 0,
               "controller_lut6": cost.controller_luts(policy), "verification": None}
    if len(patch):
        layers, one_hot_index, guard_luts = build_guard_decoder(grid_code(policy), patch.lower, patch.upper)
        words = cost.build_words(patch.deltas)
        adders = sum(cost.adder_widths(policy.output_count, words))
        summary.update(guard_lut6=guard_luts, encoder_lut6=cost.address_encoder_luts(len(patch)), adder_lut6=adders,
                       memory_bits=words.shape[1] * sum(map(cost.minimum_signed_width, words.tolist())))
        summary["total_lut6"] = guard_luts + summary["encoder_lut6"] + adders
        if verify:
            summary["verification"] = verify_patch(policy, patch, layers, one_hot_index, words)
    # Logic only on both sides: the patch's memory and the shared thermometer are excluded.
    summary["percent_of_controller"] = 100.0 * summary["total_lut6"] / summary["controller_lut6"]
    return summary
