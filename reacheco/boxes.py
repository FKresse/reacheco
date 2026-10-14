"""Closed axis-aligned boxes: splitting along the policy grid and cutting off the goal.

A collection of boxes is a tensor of shape ``(N, 2, D)`` with lower bounds in
``boxes[:, 0]`` and upper bounds in ``boxes[:, 1]``. Bounds may be infinite.
Boxes are closed, while policy cells are ``(t_{i-1}, t_i]`` (see
``reacheco.policy``).
"""
from __future__ import annotations

import torch

PIECE_BUDGET = 16384


def iter_cell_pieces(boxes, grid, piece_budget=PIECE_BUDGET):
    """Split boxes along the grid, yielding ``(pieces, cell_ids, parents)`` chunks.

    ``grid`` provides ``thresholds`` (rows ending in ``+inf``) and ``strides``,
    e.g. a ``DWCPolicy``. Each box is intersected with every cell it meets, so
    the pieces cover the box exactly. A box whose lower bound lies on a
    threshold also meets the cell left of it, in a zero-width face. Pieces
    are ordered by parent box, then by cell with the last dimension fastest.
    At most ``piece_budget`` pieces are materialized per chunk, even when a
    single box meets millions of cells.
    """
    if piece_budget < 1:
        raise ValueError("piece_budget must be positive")
    if not len(boxes):
        return
    if bool(torch.isnan(boxes).any()) or bool((boxes[:, 0] > boxes[:, 1]).any()):
        raise ValueError("boxes must not contain NaN or inverted bounds")
    uppers = [row.to(boxes) for row in grid.thresholds]
    lowers = [torch.cat((row.new_tensor([-torch.inf]), row)) for row in uppers]
    first = torch.stack([torch.searchsorted(row, boxes[:, 0, d].contiguous()) for d, row in enumerate(uppers)], 1)
    last = torch.stack([torch.searchsorted(row, boxes[:, 1, d].contiguous()) for d, row in enumerate(uppers)], 1)
    widths = last - first + 1
    ends = widths.prod(1).cumsum(0)
    starts = ends - widths.prod(1)
    radix = torch.ones_like(widths)
    for d in range(boxes.shape[-1] - 2, -1, -1):
        radix[:, d] = radix[:, d + 1] * widths[:, d + 1]
    strides = grid.strides.to(boxes.device)

    for start in range(0, int(ends[-1]), piece_budget):
        flat = torch.arange(start, min(start + piece_budget, int(ends[-1])), device=boxes.device)
        parents = torch.searchsorted(ends, flat, right=True)
        index = first[parents] + ((flat - starts[parents])[:, None] // radix[parents]) % widths[parents]
        lower = torch.maximum(boxes[parents, 0], torch.stack([row[index[:, d]] for d, row in enumerate(lowers)], 1))
        upper = torch.minimum(boxes[parents, 1], torch.stack([row[index[:, d]] for d, row in enumerate(uppers)], 1))
        yield torch.stack((lower, upper), 1), (index * strides).sum(1), parents


def cell_pieces(boxes, grid):
    """All pieces of ``iter_cell_pieces`` at once: ``(pieces, cell_ids, parents)``."""
    chunks = list(iter_cell_pieces(boxes, grid))
    if not chunks:
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        return boxes.new_empty((0, 2, boxes.shape[-1])), empty, empty
    pieces, cells, parents = zip(*chunks)
    return torch.cat(pieces), torch.cat(cells), torch.cat(parents)


def convert(boxes, dtype, *, outward):
    """Float64 boxes as ``dtype``, rounded outward (enclosing ``boxes``) or inward (enclosed by them)."""
    lower, upper = boxes[:, 0].to(dtype), boxes[:, 1].to(dtype)
    down, up = torch.full_like(lower, -torch.inf), torch.full_like(lower, torch.inf)
    if outward:
        lower = torch.where(lower.to(boxes.dtype) > boxes[:, 0], torch.nextafter(lower, down), lower)
        upper = torch.where(upper.to(boxes.dtype) < boxes[:, 1], torch.nextafter(upper, up), upper)
    else:
        lower = torch.where(lower.to(boxes.dtype) < boxes[:, 0], torch.nextafter(lower, up), lower)
        upper = torch.where(upper.to(boxes.dtype) > boxes[:, 1], torch.nextafter(upper, down), upper)
    return torch.stack((lower, upper), 1)


def inside_any(boxes, goals):
    """Whether each box is contained in at least one of the boxes ``goals`` ``(G, 2, D)``."""
    return ((boxes[:, None, 0] >= goals[None, :, 0]).all(-1)
            & (boxes[:, None, 1] <= goals[None, :, 1]).all(-1)).any(-1)


def cut_goal(boxes, goals, eligible):
    """Remove the terminal goal from the ``eligible`` boxes.

    Returns ``(pieces, parents, slabs, reached)``. The complement of a goal box
    is split into ``2 D`` convex slabs (below and above it in each dimension),
    so taking hulls only within one slab can never refill the goal. Slab
    labels combine across several goals into one label per convex region; an
    ineligible box keeps label ``0`` and stays intact (callers pass boxes whose
    incoming transition is not certified safe as ineligible). Faces shared with
    the goal stay in the pieces, so the cut over-approximates. ``reached``
    marks input boxes that lost a part to the goal.
    """
    parents = torch.arange(len(boxes), device=boxes.device)
    reached = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    pieces, slabs = boxes, torch.zeros_like(parents)
    labels_per_goal = 1 + 2 * boxes.shape[-1]
    for goal in goals:
        pieces, piece_parents, piece_slabs, piece_reached = _cut_one(pieces, goal, eligible[parents])
        reached[parents[piece_reached]] = True
        slabs = slabs[piece_parents] * labels_per_goal + piece_slabs
        parents = parents[piece_parents]
    return pieces, parents, slabs, reached


def _cut_one(boxes, goal, eligible):
    """Cut eligible boxes at one goal box: ``(pieces, parents, slabs, reached)``."""
    # eligible is used to keep unsafe transitions, so they are noticed.
    reached = eligible & (boxes[:, 0] <= goal[1]).all(-1) & (boxes[:, 1] >= goal[0]).all(-1)
    ids = torch.arange(len(boxes), device=boxes.device)
    chunks, owners, slabs = [boxes[~eligible]], [ids[~eligible]], [torch.zeros_like(ids[~eligible])]
    remainder, ids = boxes[eligible].clone(), ids[eligible]
    for d in range(boxes.shape[-1]):
        for upper in (False, True):
            boundary = goal[int(upper), d]
            outside = remainder[:, 1, d] > boundary if upper else remainder[:, 0, d] < boundary
            piece = remainder[outside].clone()
            if upper:
                piece[:, 0, d] = torch.maximum(piece[:, 0, d], boundary)
                remainder[:, 1, d] = torch.minimum(remainder[:, 1, d], boundary)
            else:
                piece[:, 1, d] = torch.minimum(piece[:, 1, d], boundary)
                remainder[:, 0, d] = torch.maximum(remainder[:, 0, d], boundary)
            chunks.append(piece)
            owners.append(ids[outside])
            # slab label, so that merging is handled correctly 
            # (merging could reintroduce the goal region)
            slabs.append(torch.full_like(ids[outside], 1 + 2 * d + int(upper)))
            # should not happen as we test for goal inclusion on top
            valid = (remainder[:, 0] <= remainder[:, 1]).all(-1)
            remainder, ids = remainder[valid], ids[valid]
    return torch.cat(chunks), torch.cat(owners), torch.cat(slabs), reached
