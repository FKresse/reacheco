"""The trainable DWC behind a hard ACC controller, for supervised fine-tuning.

The thermometer thresholds come from the hard controller
(``checkpoints/acc/dwc_b81_seed<s>.pt``), the LUT network from its training
checkpoint (``checkpoints/acc/training/dwc_b81_seed<s>.pt``, converted by
``ReachECO-dev/tools/convert_training_checkpoint.py``). Training uses
``torch_dwn``'s LUT layers (EFD backward, CUDA only); ``to_policy`` exports
the hard controller as a ``reacheco.policy.DWCPolicy``.
"""
from __future__ import annotations

import torch
import torch_dwn as dwn
from torch import nn

from reacheco.policy import FORMAT, VERSION, DWCPolicy


class Head(nn.Module):
    """Popcounts to actions: ``exp(log_alpha) * (popcount / N - 0.5) + beta`` (unclipped)."""

    def __init__(self, outputs, action_dim):
        super().__init__()
        self.outputs = outputs // action_dim
        self.log_alpha = nn.Parameter(torch.zeros(action_dim))
        self.beta = nn.Parameter(torch.zeros(action_dim))

    def forward(self, popcounts):
        return torch.exp(self.log_alpha) * (popcounts / self.outputs - 0.5) + self.beta


class TrainableDWC(nn.Module):
    """Thermometer, two LUT layers (the first with a learnable mapping), popcount and head."""

    def __init__(self, hard, checkpoint, device="cuda"):
        super().__init__()
        state, n = checkpoint["network"], checkpoint["lut_inputs"]
        self.register_buffer("thresholds", torch.stack([row[:-1] for row in hard.thresholds]).cpu())
        self.register_buffer("lower", checkpoint["lower"].float())
        self.register_buffer("upper", checkpoint["upper"].float())
        if not (torch.equal(self.lower, hard.lower.cpu()) and torch.equal(self.upper, hard.upper.cpu())):
            raise ValueError("the training checkpoint does not belong to this controller")
        inputs, hidden = self.thresholds.numel(), len(state["net.1.luts"])
        outputs = len(state["net.2.luts"])
        self.net = nn.Sequential(
            nn.Flatten(),
            dwn.LUTLayer(inputs, hidden, n=n, mapping="learnable"),
            dwn.LUTLayer(hidden, outputs, n=n, mapping=state["net.2.mapping"].int()),
            dwn.GroupSum(k=hard.action_dim, tau=1.0),
            Head(outputs, hard.action_dim),
        )
        self.net.load_state_dict({key[len("net."):]: value for key, value in state.items()})
        self.hard = hard
        self.to(device)

    def forward(self, states):
        """Unclipped actions of states ``(N, D)``: the regression target ignores the action clip."""
        states = torch.maximum(torch.minimum(states, self.upper), self.lower)
        return self.net((states[..., None] > self.thresholds).float())

    def to_policy(self):
        """The hard controller: argmax mapping, LUT signs, no overrides."""
        first, second, head = self.net[1], self.net[2], self.net[4]
        mapping = first.mapping.weights.detach().argmax(dim=0).reshape(first.output_size, -1)
        artifact = self.hard.to_artifact()
        artifact["network"].update(
            mappings=[mapping.long().cpu(), second.mapping.detach().long().cpu()],
            luts=[(layer.luts.detach() > 0).cpu() for layer in (first, second)],
            log_alpha=head.log_alpha.detach().float().cpu(), beta=head.beta.detach().float().cpu())
        artifact["overrides"] = {"cell_ids": torch.zeros(0, dtype=torch.long),
                                 "values": torch.zeros((0, self.hard.action_dim), dtype=torch.long)}
        assert artifact["format"] == FORMAT and artifact["version"] == VERSION
        return DWCPolicy(artifact, device=self.hard.device)


def load(hard_path, checkpoint_path, device="cuda"):
    """The trainable DWC of a hard controller and its training checkpoint."""
    return TrainableDWC(DWCPolicy.load(hard_path, device=device),
                        torch.load(checkpoint_path, map_location="cpu", weights_only=True), device=device)
