"""A minimal netlist: layers of small look-up tables, in plain tensors.

Truth tables follow the ``torch_dwn`` convention (``2 ** k`` entries in
``{-1, +1}``, positive meaning one), but nothing here needs ``torch_dwn`` or a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: LUT2 truth tables, indexed by ``a + 2 * b``.
AND_2 = (-1, -1, -1, 1)       # a AND b            (also identity when b is a)
AND_NOT = (-1, 1, -1, -1)     # a AND NOT b
NOT_1 = (1, -1, -1, -1)       # NOT a              (when both inputs are a)
TRUE = (1, 1, 1, 1)           # constant one


@dataclass
class LutLayer:
    """One layer of ``k``-input look-up tables.

    ``mapping[i]`` lists the indices of the layer inputs feeding LUT ``i`` and
    ``luts[i]`` is its truth table, addressed by
    ``sum_j input[mapping[i, j]] << j``.
    """

    input_size: int
    mapping: torch.Tensor      # (gates, k), long
    luts: torch.Tensor         # (gates, 2 ** k), int8 in {-1, +1}

    def __post_init__(self):
        self.input_size = int(self.input_size)
        self.mapping = torch.as_tensor(self.mapping, dtype=torch.long).reshape(len(self.luts), -1)
        self.luts = torch.as_tensor(self.luts, dtype=torch.int8)
        if self.luts.shape[1] != 2 ** self.fan_in:
            raise ValueError(f"a {self.fan_in}-input LUT needs {2 ** self.fan_in} truth-table entries")
        if self.input_size and bool(((self.mapping < 0) | (self.mapping >= self.input_size)).any()):
            raise ValueError("LUT mapping refers to an input outside the layer")

    @property
    def output_size(self) -> int:
        return int(self.mapping.shape[0])

    @property
    def fan_in(self) -> int:
        return int(self.mapping.shape[1])

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate the layer on a ``(batch, input_size)`` boolean tensor."""
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(f"expected (batch, {self.input_size}) inputs, got {tuple(values.shape)}")
        device = values.device
        powers = 2 ** torch.arange(self.fan_in, device=device)
        address = (values[:, self.mapping.to(device)].long() * powers).sum(-1)
        gate = torch.arange(self.output_size, device=device)[None]
        return self.luts.to(device)[gate, address] > 0


def evaluate(layers, values: torch.Tensor) -> torch.Tensor:
    """Run a boolean input through a chain of layers."""
    for layer in layers:
        values = layer(values)
    return values


def and_table(fan_in: int) -> torch.Tensor:
    """Truth table of a ``fan_in``-input AND: true only when every input is one."""
    table = torch.full((2 ** int(fan_in),), -1, dtype=torch.int8)
    table[-1] = 1
    return table
