"""Resource model of the patch and of the controller it patches, in LUT6.

The patch sits between the controller's popcount and its decoder::

    thermometer bits -> guard decoder -> one-hot -> address -> memory[action] -> + popcount

Address 0 holds a zero delta, so a state outside every repaired cell keeps the
network's value; each cluster occupies one word per action dimension. Counts
are technology independent and pessimistic: nothing is shared or packed.
"""
from __future__ import annotations

import math

import torch


def minimum_signed_width(values) -> int:
    """Narrowest two's-complement width holding every value."""
    low, high = int(min(values)), int(max(values))
    width = 1
    while low < -(1 << (width - 1)) or high > (1 << (width - 1)) - 1:
        width += 1
    return width


def unsigned_width(maximum: int) -> int:
    """Bits for the unsigned range ``0 .. maximum``."""
    return max(1, int(maximum).bit_length())


def build_words(deltas: torch.Tensor) -> torch.Tensor:
    """Memory contents ``(actions, clusters + 1)``; address 0 is the zero delta."""
    deltas = torch.as_tensor(deltas, dtype=torch.long)
    return torch.cat((deltas.new_zeros((1, deltas.shape[1])), deltas)).transpose(0, 1).contiguous()


def or_tree_luts(fan_in: int, lut_inputs: int = 6) -> int:
    """LUTs combining ``fan_in`` signals into one: ``ceil((f - 1) / (k - 1))``."""
    return 0 if fan_in <= 1 else math.ceil((fan_in - 1) / (lut_inputs - 1))


def address_encoder_luts(clusters: int) -> int:
    """One-hot to binary address: one OR tree per address bit; cluster ``c`` drives address ``c + 1``."""
    width = max(1, math.ceil(math.log2(clusters + 1)))
    return sum(or_tree_luts(sum(((c + 1) >> bit) & 1 for c in range(clusters))) for bit in range(width))


def adder_widths(popcount_max: int, words: torch.Tensor) -> list:
    """Width of each action's ``popcount + delta`` adder, from the range of the sum."""
    return [minimum_signed_width((int(row.min()), popcount_max + int(row.max()))) for row in words]


def popcount_luts(inputs: int) -> float:
    """LUT6 of an ``N``-input popcount: the full and half adders of Bacellar et al.
    (Differentiable Weightless Neural Networks, appendix F), ``N - log2(N/3) - 2``
    and ``N/3 - 1``, one dual-output LUT6 each."""
    if inputs <= 2:
        return float(max(0, inputs - 1))
    return (inputs - math.log2(inputs / 3.0) - 2.0) + (inputs / 3.0 - 1.0)


def controller_luts(policy) -> float:
    """The unpatched controller's logic: LUT layers plus one popcount per action.

    The thermometer encoding is shared by every variant of a controller and is
    excluded, as is the patch's memory.
    """
    return sum(len(mapping) for mapping in policy.mappings) + policy.action_dim * popcount_luts(policy.output_count)


def rom_luts(depth: int, width: int) -> int:
    """LUT6 of a ``depth`` x ``width`` table held in logic: one 64 x 1 LUT6 ROM per 64 words and
    output bit, plus a multiplexer tree over the slices (one LUT-equivalent per extra slice)."""
    slices = max(1, math.ceil(2 ** max(1, math.ceil(math.log2(depth))) / 64))
    return width * (2 * slices - 1)
