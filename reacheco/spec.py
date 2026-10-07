"""Safety specifications and verification problems.

A ``SafetySpec`` fixes the horizon, the initial set ``X0``, the terminal goal
set and the unsafe set; the dynamics belong to the ``System``. A
``Problem`` pairs the two. Sets are unions of closed boxes ``(N, 2, D)``,
stored as float64 CPU tensors.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch


@dataclass(frozen=True, eq=False)
class SafetySpec(ABC):
    """Avoid the unsafe set for ``horizon`` control steps or until safe goal arrival.

    A trajectory that enters the goal with a safe incoming transition has
    completed the task and is not followed further. An empty goal set means
    pure bounded safety.
    """

    horizon: int
    initial_boxes: torch.Tensor
    goal_boxes: torch.Tensor
    state_dim: ClassVar[int]

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 0:
            raise ValueError("horizon must be a nonnegative integer number of control steps")
        for name, finite in (("initial_boxes", True), ("goal_boxes", False)):
            boxes = torch.as_tensor(getattr(self, name), dtype=torch.float64).reshape(-1, 2, self.state_dim)
            if bool(torch.isnan(boxes).any()) or bool((boxes[:, 0] > boxes[:, 1]).any()):
                raise ValueError(f"{name} must not contain NaN or inverted bounds")
            if finite and not bool(torch.isfinite(boxes).all()):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, boxes)

    @abstractmethod
    def intersects_unsafe(self, boxes):
        """Per box: ``False`` certifies that the box avoids the unsafe set."""

    @abstractmethod
    def inside_unsafe(self, boxes):
        """Per box: ``True`` certifies that the whole box is unsafe."""

    @classmethod
    def from_dict(cls, data, system, initial_boxes, goal_boxes):
        """Build the specification from its problem-file entries."""
        return cls(int(data["horizon"]), initial_boxes, goal_boxes)


@dataclass(frozen=True, eq=False)
class Problem:
    system: object # defined in enviroments/
    spec: SafetySpec
    path: Path | None = None


def load_problem(path):
    """Load a problem file.

    Format::

        {"environment": "acc", "options": {<system config>},
         "horizon": 200,
         "initial_set": {"file": "sets/x0.json", "clip": {"dimension": 0, "upper": 10.0}},
         "goal_set": {"file": "sets/goal.json"},
         <environment-specific specification entries>}

    Box files contain ``{"boxes": [[lower, upper], ...]}``; paths are relative
    to the problem file. ``clip`` keeps the part of the set within
    ``[lower, upper]`` along one dimension.
    """
    from reacheco.environments import ENVIRONMENTS

    path = Path(path).resolve()
    data = json.loads(path.read_text())
    system_class, spec_class = ENVIRONMENTS[data["environment"]]
    system = system_class(system_class.config_class(**data.get("options", {})))
    # both inital and goal are represented by boxes.
    initial = _load_boxes(data["initial_set"], path.parent, spec_class.state_dim)
    goal = _load_boxes(data.get("goal_set"), path.parent, spec_class.state_dim)
    return Problem(system, spec_class.from_dict(data, system, initial, goal), path)


def _load_boxes(entry, directory, dimension):
    if entry is None:
        return torch.empty((0, 2, dimension), dtype=torch.float64)
    boxes = torch.tensor(json.loads((directory / entry["file"]).read_text())["boxes"],
                         dtype=torch.float64).reshape(-1, 2, dimension)
    clip = entry.get("clip")
    if clip is not None:
        d = clip["dimension"]
        lower, upper = clip.get("lower", -torch.inf), clip.get("upper", torch.inf)
        boxes = boxes[(boxes[:, 1, d] > lower) & (boxes[:, 0, d] < upper)].clone()
        boxes[:, 0, d] = boxes[:, 0, d].clamp(min=lower)
        boxes[:, 1, d] = boxes[:, 1, d].clamp(max=upper)
    return boxes
