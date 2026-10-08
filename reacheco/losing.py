"""The losing set ``L``: a certified under-approximation of the states from which
no continuation (of the chosen backend) avoids the unsafe set ``U``.

``L`` always contains ``U``, which stays implicit in the specification; only
the boxes certified beyond it are stored. They are split along the policy
cells and kept per cell on the CPU; queries run on the query's device.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from reacheco.boxes import cell_pieces

PAIR_BATCH = 250_000


class LosingSet:
    """Monotone set of certified losing boxes, queried per policy cell."""

    def __init__(self, problem, policy, boxes=None):
        self.problem, self.policy = problem, policy
        self._dtype = policy.thresholds[0].dtype
        self._dimension = len(policy.thresholds)
        self._cells = {}
        self._device_tables = {}
        self.version = 0
        if boxes is not None and len(boxes):
            self.add(boxes)

    @classmethod
    def load(cls, problem, policy, path):
        """``L`` from a box file ``{"boxes": [[lower, upper], ...]}``, split along this policy's cells."""
        return cls(problem, policy, json.loads(Path(path).read_text())["boxes"])

    def __len__(self):
        return sum(len(boxes) for boxes in self._cells.values())

    @property
    def boxes(self):
        if not self._cells:
            return torch.empty((0, 2, self._dimension), dtype=self._dtype)
        return torch.cat([self._cells[cell] for cell in sorted(self._cells)])

    def clone(self):
        result = LosingSet(self.problem, self.policy)
        result._cells = {cell: boxes.clone() for cell, boxes in self._cells.items()}
        result.version = self.version
        return result

    # ------------------------------------------------------------- queries

    def intersects(self, boxes, cells):
        """Per policy-aligned box with its cell: does it meet ``L`` (``U`` or a stored box)?"""
        hit = self.problem.spec.intersects_unsafe(boxes)
        return hit | self._compare(boxes, cells, contained=False) if self._cells else hit

    def contains(self, boxes, cells):
        """Per policy-aligned box: is it inside ``U`` or inside one stored box of its cell?

        A box covered only by several stored boxes together is reported as
        not contained; that only weakens pruning, never soundness.
        """
        inside = self.problem.spec.inside_unsafe(boxes)
        return inside | self._compare(boxes, cells, contained=True) if self._cells else inside

    def _compare(self, boxes, cells, *, contained):
        """Compare each box with the stored boxes of its cell, pairing only same-cell boxes."""
        result = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        if not len(boxes):
            return result
        occupied, counts, offsets, stored = self._table(boxes.device, boxes.dtype)
        cells = torch.as_tensor(cells, dtype=torch.long, device=boxes.device)
        position = torch.searchsorted(occupied, cells).clamp(max=len(occupied) - 1)
        known = occupied[position] == cells
        count = torch.where(known, counts[position], 0)
        offset = offsets[position]
        ends = count.cumsum(0)
        for start in range(0, int(ends[-1]), PAIR_BATCH):
            pairs = torch.arange(start, min(start + PAIR_BATCH, int(ends[-1])), device=boxes.device)
            rows = torch.searchsorted(ends, pairs, right=True)
            other = stored[offset[rows] + pairs - (ends[rows] - count[rows])]
            query = boxes[rows]
            if contained:
                match = ((other[:, 0] <= query[:, 0]) & (other[:, 1] >= query[:, 1])).all(1)
            else:
                match = ((query[:, 0] <= other[:, 1]) & (query[:, 1] >= other[:, 0])).all(1)
            result[rows[match]] = True
        return result

    def _table(self, device, dtype):
        """Occupied cells (sorted), box counts and offsets, and all stored boxes, cached per device."""
        key = (str(device), dtype)
        cached = self._device_tables.get(key)
        if cached is None or cached[0] != self.version:
            occupied = sorted(self._cells)
            counts = torch.tensor([len(self._cells[cell]) for cell in occupied], dtype=torch.long)
            cached = (self.version, torch.tensor(occupied, dtype=torch.long, device=device), counts.to(device),
                      (counts.cumsum(0) - counts).to(device),
                      torch.cat([self._cells[cell] for cell in occupied]).to(device=device, dtype=dtype))
            self._device_tables[key] = cached
        return cached[1:]

    # -------------------------------------------------------------- growth

    def add(self, boxes):
        """Store certified losing boxes; returns the pieces that enlarged ``L``.

        Boxes are split along the policy cells. In each cell a piece already
        covered by the stored boxes is skipped; the rest are stored, boxes
        inside others are dropped, neighbours forming a box are merged, and a
        cell covered completely is stored as one box.
        """
        boxes = torch.as_tensor(boxes, dtype=self._dtype).reshape(-1, 2, self._dimension)
        if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 0] > boxes[:, 1]).any()):
            raise ValueError("losing boxes must be finite and not inverted")
        pieces, cells, _ = cell_pieces(boxes.cpu(), self.policy)
        grouped = {}
        for piece, cell in zip(pieces, cells.tolist()):
            grouped.setdefault(cell, []).append(piece)
        added = []
        for cell, candidates in grouped.items():
            stored = self._cells.get(cell, torch.empty((0, 2, self._dimension), dtype=self._dtype))
            accepted = []
            for candidate in candidates:
                current = torch.cat((stored, torch.stack(accepted))) if accepted else stored
                if not _covers(candidate, current):
                    accepted.append(candidate)
            if not accepted:
                continue
            added.extend(accepted)
            combined = _merge_adjacent(_drop_contained(torch.cat((stored, torch.stack(accepted)))))
            whole = self.cell_box(cell)
            self._cells[cell] = whole[None] if whole is not None and _covers(whole, combined) else combined
        if not added:
            return torch.empty((0, 2, self._dimension), dtype=self._dtype)
        self.version += 1
        self._device_tables.clear()
        return torch.stack(added)

    def cell_box(self, cell):
        """The part of a policy cell inside the grid bounds, or ``None`` if there is none.

        Inner cells ``(t_{i-1}, t_i]`` are stored closed; the unbounded outer
        cells end at the grid bounds. The upper outer cell ``(t_last, inf)`` is
        open at ``t_last``, so it has no part inside the grid when ``t_last``
        is at or above the upper bound.
        """
        lower, upper = [], []
        for d, row in enumerate(self.policy.thresholds):
            index = (int(cell) // int(self.policy.strides[d])) % int(self.policy.sizes[d])
            row = row.cpu()
            low = float(self.policy.lower[d]) if index == 0 else float(row[index - 1])
            high = float(self.policy.upper[d]) if index == len(row) - 1 else float(row[index])
            if high < low or (index == len(row) - 1 and high <= low):
                return None
            lower.append(low)
            upper.append(high)
        return torch.tensor([lower, upper], dtype=self._dtype)


def _covers(target, boxes):
    """Whether the union of ``boxes`` covers ``target`` (exact sweep, recursive in the dimension)."""
    if not len(boxes):
        return False
    target, boxes = target.double().cpu(), boxes.double().cpu()
    lo, hi = target
    if bool(((boxes[:, 0] <= lo) & (boxes[:, 1] >= hi)).all(1).any()):
        return True
    # A degenerate target is only accepted by direct containment.
    if bool(((hi - lo) <= 0).any()):
        return False
    clipped_lo, clipped_hi = torch.maximum(boxes[:, 0], lo), torch.minimum(boxes[:, 1], hi)
    useful = ((clipped_hi - clipped_lo) > 0).all(1)
    clipped_lo, clipped_hi = clipped_lo[useful], clipped_hi[useful]
    if not len(clipped_lo):
        return False
    if target.shape[1] == 1:
        covered = float(lo[0])
        for start, end in sorted(zip(clipped_lo[:, 0].tolist(), clipped_hi[:, 0].tolist())):
            if end <= covered:
                continue
            if start > covered:
                return False
            covered = end
            if covered >= float(hi[0]):
                return True
        return False
    edges = torch.unique(torch.cat((lo[:1], hi[:1], clipped_lo[:, 0], clipped_hi[:, 0])), sorted=True)
    for left, right in zip(edges[:-1].tolist(), edges[1:].tolist()):
        if right - left <= 0 or right <= float(lo[0]) or left >= float(hi[0]):
            continue
        spans = (clipped_lo[:, 0] <= left) & (clipped_hi[:, 0] >= right)
        if not bool(spans.any()) or not _covers(target[:, 1:],
                                                 torch.stack((clipped_lo[spans, 1:], clipped_hi[spans, 1:]), 1)):
            return False
    return True


def _drop_contained(boxes):
    """Drop every box inside another box; of equal boxes the first is kept."""
    lower, upper = boxes[:, 0], boxes[:, 1]
    # inside[o, i]: box o contains box i; larger[o, i]: and is strictly larger somewhere.
    inside = (lower[:, None] <= lower[None]).all(-1) & (upper[:, None] >= upper[None]).all(-1)
    larger = (lower[:, None] < lower[None]).any(-1) | (upper[:, None] > upper[None]).any(-1)
    order = torch.arange(len(boxes))
    earlier = order[:, None] < order[None]
    dropped = (inside & (larger | earlier) & (order[:, None] != order[None])).any(0)
    return boxes[~dropped]


def _merge_adjacent(boxes):
    """Merge two boxes whose union is a box, until no such pair remains.

    Pairs are merged in row-major order of ``(left, right)``, ``left < right``:
    the hull replaces ``left`` and ``right`` is removed, then the search
    restarts. Two boxes form a box when they agree exactly in all but one
    dimension and touch or overlap in that one.
    """
    dimensions = boxes.shape[2]
    others = ~torch.eye(dimensions, dtype=torch.bool)
    while len(boxes) > 1:
        lower, upper = boxes[:, 0], boxes[:, 1]
        equal = (lower[:, None] == lower[None]) & (upper[:, None] == upper[None])          # (n, n, D)
        agree = (equal[:, :, None, :] | ~others).all(-1)                                   # (n, n, D): all others equal
        touch = (lower[:, None] <= upper[None]) & (lower[None] <= upper[:, None])          # (n, n, D)
        pairs = torch.nonzero(torch.triu((agree & touch).any(-1), diagonal=1))
        if not len(pairs):
            break
        left, right = pairs[0].tolist()
        merged = torch.stack((torch.minimum(boxes[left, 0], boxes[right, 0]),
                              torch.maximum(boxes[left, 1], boxes[right, 1])))
        boxes = torch.cat((boxes[:left], merged[None], boxes[left + 1:right], boxes[right + 1:]))
    return boxes
