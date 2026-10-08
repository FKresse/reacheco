"""Certify the intersample part of SafeStep: ``Tube(B, a)`` avoids the unsafe set ``U``.

A false certificate means "not certified safe": it marks an action as unusable
for the box, but never shows that the box is losing.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def tube_safe(problem, boxes, actions, *, segments=None, box_budget=250_000, batch_size=32_768):
    """Per box ``(N, 2, D)`` and held action ``(N, action_dim)``: does ``Tube(B, a)`` avoid ``U``?

    The whole control period is checked as one segment first. Boxes that fail
    are re-enclosed with each segment count of ``segments`` in turn (default:
    the problem's ``tube_segments``); every level only processes the boxes the
    previous one rejected. Before the second level, boxes that are unsafe
    themselves or whose ``Post`` is unsafe are given up: their tube meets ``U``
    at every segmentation. At most ``box_budget`` tube segments and
    ``batch_size`` boxes are enclosed at a time.
    """
    levels = [int(level) for level in (problem.tube_segments if segments is None else segments) if int(level) > 1]
    if any(later <= earlier for earlier, later in zip(levels, levels[1:])):
        raise ValueError("segment counts must increase")
    safe = torch.empty(len(boxes), dtype=torch.bool, device=boxes.device)
    chunk = min(batch_size, box_budget)
    for start in range(0, len(boxes), chunk):
        box, action = boxes[start:start + chunk], actions[start:start + chunk]
        certified = ~_meets_unsafe(problem, problem.system.tube(box, action, 1))
        for index, level in enumerate(levels):
            retry = torch.where(~certified)[0]
            if index == 1:
                hopeless = (problem.spec.intersects_unsafe(box[retry])
                            | problem.spec.intersects_unsafe(problem.system.post(box[retry], action[retry])))
                retry = retry[~hopeless]
            if not len(retry):
                break
            for part in retry.split(min(batch_size, max(1, box_budget // level))):
                certified[part] = ~_meets_unsafe(problem, problem.system.tube(box[part], action[part], level))
        safe[start:start + chunk] = certified
    return safe


def _meets_unsafe(problem, tubes):
    """Whether any segment of each tube ``(S, N, 2, D)`` intersects ``U``."""
    segments, count = tubes.shape[:2]
    return problem.spec.intersects_unsafe(tubes.reshape(segments * count, *tubes.shape[2:])).reshape(
        segments, count).any(0)
