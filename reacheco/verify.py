"""Forward closed-loop verification without repair (Algorithm 1 without BRepair)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from reacheco.reach import detect_fixed_point, reach_step


@dataclass
class Verification:
    """Outcome of a forward pass.

    ``certified`` with ``horizon`` ``T``, or ``inf`` for an inductive fixed
    point or when every trajectory has safely entered the terminal goal.
    Otherwise ``failure_step`` is the first step with failing rows
    ``failing`` (a mask over ``frontiers[failure_step]``).
    """

    certified: bool
    horizon: float | None
    frontiers: list = field(repr=False)
    records: list = field(repr=False)
    failure_step: int | None = None
    failing: torch.Tensor | None = field(default=None, repr=False)


def verify(problem, policy, meets_losing=None, *, merge=True, merge_disabled=(), merge_disabled_at=None):
    """Propagate ``X0`` until a failure, the goal, the horizon or (if the
    specification asks for it) an inductive fixed point.

    ``meets_losing(boxes, cells)`` defaults to the unsafe set ``U``.
    ``merge_disabled`` cells are never merged; ``merge_disabled_at[t]`` adds
    cells for the step producing ``X_t``.
    """
    if meets_losing is None:
        def meets_losing(boxes, cells):
            return problem.spec.intersects_unsafe(boxes)
    merge_disabled_at = merge_disabled_at or {}
    boxes = problem.spec.initial_boxes.to(device=policy.device, dtype=torch.float32)
    frontiers, records = [boxes.cpu()], []
    for step in range(1, problem.spec.horizon + 1):
        if not len(boxes):
            # Every trajectory has entered the goal with a safe transition.
            return Verification(True, math.inf, frontiers, records)
        disabled = set(merge_disabled) | set(merge_disabled_at.get(step, ()))
        frontier, record = reach_step(boxes, policy, problem, meets_losing, merge=merge,
                                      merge_disabled=disabled, previous=records[-1] if records else None)
        frontiers.append(frontier.cpu())
        records.append(record)
        if bool(record.failing.any()):
            return Verification(False, None, frontiers, records, step, record.failing)
        if (problem.spec.detect_fixed_point and len(records) >= 2
                and detect_fixed_point(boxes, records[-2].cells.to(boxes.device), frontier,
                                       record.cells.to(frontier.device))):
            return Verification(True, math.inf, frontiers, records)
        boxes = frontier
    return Verification(True, float(problem.spec.horizon), frontiers, records)
