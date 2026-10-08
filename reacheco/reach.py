"""ReachStep: one step of closed-loop reachability, merging boxes per policy cell.

A step from ``X_k`` to ``X_{k+1}``:

1. remove the terminal goal from ``X_k`` and split it along the policy cells
   into *source boxes*; a source box lies in a single policy cell ``q(B)``,
   so the controller applies one action ``pi(q(B))`` to all of it,
2. bound every source box's transition with ``Post`` and ``Tube``,
3. split the post-images along the policy cells into *pieces* and remove the
   goal again,
4. merge the pieces of each cell into their hull, one *row* of ``X_{k+1}``.

Pieces stay separate in merge-disabled cells and wherever merging would
create a failure no piece has: a hull meeting the losing set ``L`` while none
of its pieces does, or (lookahead) a hull whose own next transition fails
while every piece's passes. The record keeps all pieces and their source
boxes, so a merged row lists every candidate parent. ("Source box" follows
Appendix D, assumption 2: ReachStep reports every source box whose tube meets
``U`` or whose post-image meets ``L``.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from reacheco.boxes import cell_pieces, cut_goal, iter_cell_pieces
from reacheco.tubes import tube_safe

TRANSITION_BATCH = 32_768


def successors(problem, policy, boxes, actions, tube_ok, *, piece_budget=None):
    """``Succ(B, a)`` of policy-aligned boxes held at ``actions``, in chunks.

    Yields ``(pieces, cells, parents, slabs, reached)``: ``Post`` split along
    the policy cells, with the goal removed from every piece whose transition
    is safe (``tube_ok`` for its parent box and the piece outside ``U``); the
    goal-complement slab of each piece (see ``cut_goal``); and the parents that
    lost a part to the goal. Forward steps and continuation games both use
    this, so they share one transition semantics. ``piece_budget=None`` yields
    a single chunk; otherwise at most that many pieces per chunk.
    """
    spec = problem.spec
    goals = spec.goal_boxes.to(boxes)
    posts = problem.system.post(boxes, actions)
    chunks = [cell_pieces(posts, policy)] if piece_budget is None else iter_cell_pieces(posts, policy, piece_budget)
    for pieces, cells, parents in chunks:
        pieces, kept, slabs, reached = cut_goal(pieces, goals, tube_ok[parents] & ~spec.intersects_unsafe(pieces))
        yield pieces, cells[kept], parents[kept], slabs, parents[reached]


@dataclass(frozen=True)
class StepRecord:
    """What later stages need to know about one transition ``X_k -> X_{k+1}``.

    All tensors live on the CPU.

    Source boxes, the policy-aligned parts of ``X_k`` (``M`` rows):
        ``source_boxes`` ``(M, 2, D)``, ``source_parents`` (row of ``X_k``),
        ``source_cells``, ``source_values`` (native values applied) and
        ``source_tube_safe``.
    Pieces of the post-images (``P`` rows): ``piece_sources`` (the source box
        producing the piece) and ``piece_rows`` (the row of ``X_{k+1}``
        containing it).
    Rows of ``X_{k+1}`` (``R`` rows): ``cells``; ``tube_unsafe`` when some
        incoming tube is not certified; ``failing``, the rows violating
        SafeStep (``tube_unsafe`` or the row meeting ``L``), i.e. the failing
        boxes ``F`` of Algorithm 1; ``row_sources``, the source box of a row
        made of a single piece and ``-1`` for a merged row.
    ``merged_pieces`` and ``merged_piece_rows``: the pieces of merged rows,
        used to certify tubes piecewise in the next step.
    ``touched_cells``: every policy cell whose action the step read.
    """

    source_boxes: torch.Tensor
    source_parents: torch.Tensor
    source_cells: torch.Tensor
    source_values: torch.Tensor
    source_tube_safe: torch.Tensor
    piece_sources: torch.Tensor
    piece_rows: torch.Tensor
    cells: torch.Tensor
    tube_unsafe: torch.Tensor
    failing: torch.Tensor
    row_sources: torch.Tensor
    merged_pieces: torch.Tensor # originial pieces before merging
    merged_piece_rows: torch.Tensor # which merged box/row now contains that piece
    touched_cells: torch.Tensor


@torch.no_grad()
def reach_step(boxes, policy, problem, meets_losing, *, merge=True, merge_disabled=(), previous=None,
               lookahead=True):
    """Propagate ``X_k`` (``boxes``) one step; returns ``X_{k+1}`` and its ``StepRecord``.

    ``meets_losing(boxes, cells)`` tells which policy-aligned boxes intersect
    ``L``. ``merge=False`` keeps every piece as its own row. ``previous`` is the
    record that produced ``boxes``; with it, a source box inside a merged row
    whose tube fails is certified piece by piece instead.
    """
    spec = problem.spec
    goals = spec.goal_boxes.to(boxes)
    active, active_parents, _, _ = cut_goal(boxes, goals, ~spec.intersects_unsafe(boxes))
    source_boxes, source_cells, parents = cell_pieces(active, policy)
    source_parents = active_parents[parents]

    # compute actions for non-goal boxes
    values = policy.cell_values(source_cells)
    actions = policy.decode(values)
    # check the tubes of the merged boxes
    tube_ok = tube_safe(problem, source_boxes, actions)
    # for failing tubes, check their unmerged pieces, if safe piecewise tube we can continue
    # failing post images are flagged at the end of the step (``failing``).
    # TODO! ablate if this actually helps on the benchmarks
    tube_ok = _certify_tubes_piecewise(problem, source_boxes, source_parents, actions, tube_ok, previous)
    # One chunk: the goal is cut over all pieces at once, which fixes the row order.
    # slabs (integer index tensor) decide if boxes are merge eligible. 
    # This is necessary (instead of only a simple check of q in Q)
    # because of goal cutting.
    pieces, piece_cells, piece_sources, slabs, _ = next(
        successors(problem, policy, source_boxes, actions, tube_ok))
    touched = [source_cells]

    if not merge or not len(pieces):
        rows = torch.arange(len(pieces), device=pieces.device)
        cells, frontier = piece_cells, pieces
    else:
        # Pieces of one cell and one goal-complement slab form a group; the
        # due to slabs a group hull cannot refill the goal.
        slab_count = max(int(slabs.max()) + 1, 2 * pieces.shape[-1] + 1)
        _, group = torch.unique(piece_cells * slab_count + slabs, sorted=True, return_inverse=True)
        group_cells = torch.zeros(int(group.max()) + 1, dtype=torch.long, device=pieces.device)
        group_cells.scatter_(0, group, piece_cells)
        # collect where merging is prohibited due to merge restirctions
        disabled = torch.as_tensor(sorted(int(c) for c in merge_disabled), dtype=torch.long, device=pieces.device)
        separate = torch.isin(piece_cells, disabled)
        # compute indices of the merged frontier and their cell idxs.
        rows, cells = _assign_rows(group, group_cells, piece_cells, separate)
        # check if the merging leads to the merged boxes being part of the losing set
        separate |= _merge_induced(pieces, piece_cells, rows, len(cells), meets_losing)
        rows, cells = _assign_rows(group, group_cells, piece_cells, separate)
        frontier = _hulls(pieces, rows, len(cells))
        if lookahead:
            # above we check if merging immediately leads to a failure. 
            # here we additionally  check if merging leads to a failure in the next transition.
            merged_rows = torch.where(torch.bincount(rows, minlength=len(cells)) > 1)[0]
            if len(merged_rows):
                touched.append(cells[merged_rows])
                failing_rows = merged_rows[_transition_fails(problem, policy, frontier[merged_rows],
                                                             cells[merged_rows], meets_losing)]
                if len(failing_rows):
                    member = torch.isin(rows, failing_rows)
                    touched.append(piece_cells[member])
                    piece_fails = _transition_fails(problem, policy, pieces[member], piece_cells[member],
                                                    meets_losing)
                    genuine = torch.zeros(len(cells), dtype=torch.bool, device=pieces.device)
                    genuine[rows[member][piece_fails]] = True
                    split = failing_rows[~genuine[failing_rows]]
                    if len(split):
                        separate |= torch.isin(rows, split)
                        rows, cells = _assign_rows(group, group_cells, piece_cells, separate)
                        frontier = _hulls(pieces, rows, len(cells))

    counts = torch.bincount(rows, minlength=len(cells))
    first_piece = torch.full((len(cells),), -1, dtype=torch.long, device=pieces.device)
    first_piece.scatter_reduce_(0, rows, torch.arange(len(pieces), device=pieces.device), reduce="amin",
                                include_self=False)
    row_sources = torch.where(counts == 1, piece_sources[first_piece.clamp(min=0)], -1)
    unsafe_tubes = torch.zeros(len(cells), dtype=torch.long, device=pieces.device)
    unsafe_tubes.index_add_(0, rows, (~tube_ok[piece_sources]).long())
    merged = counts[rows] > 1
    # A merged row meets L exactly when one of its pieces does (merge-induced
    # hits were split above), so checking rows is checking every Post piece.
    failing = (unsafe_tubes > 0) | meets_losing(frontier, cells)
    record = StepRecord(
        source_boxes=source_boxes.cpu(), source_parents=source_parents.cpu(),
        source_cells=source_cells.cpu(), source_values=values.cpu(), source_tube_safe=tube_ok.cpu(),
        piece_sources=piece_sources.cpu(), piece_rows=rows.cpu(), cells=cells.cpu(),
        tube_unsafe=(unsafe_tubes > 0).cpu(), failing=failing.cpu(), row_sources=row_sources.cpu(),
        merged_pieces=pieces[merged].cpu(), merged_piece_rows=rows[merged].cpu(),
        touched_cells=torch.unique(torch.cat(touched)).cpu(),
    )
    return frontier, record


def _assign_rows(group, group_cells, piece_cells, separate):
    """Rows of ``X_{k+1}``: one per mergeable group (in group order), then one per separate piece."""
    group_separate = torch.zeros(len(group_cells), dtype=torch.bool, device=group.device)
    group_separate[group[separate]] = True
    separate = group_separate[group]
    mergeable = torch.where(~group_separate)[0]
    row_of_group = torch.full((len(group_cells),), -1, dtype=torch.long, device=group.device)
    row_of_group[mergeable] = torch.arange(len(mergeable), device=group.device)
    rows = row_of_group[group]
    rows[separate] = len(mergeable) + torch.arange(int(separate.sum()), device=group.device)
    return rows, torch.cat((group_cells[mergeable], piece_cells[separate]))


def _hulls(pieces, rows, count):
    lower = torch.full((count, pieces.shape[-1]), torch.inf, dtype=pieces.dtype, device=pieces.device)
    upper = torch.full_like(lower, -torch.inf)
    index = rows[:, None].expand(-1, pieces.shape[-1])
    lower.scatter_reduce_(0, index, pieces[:, 0], reduce="amin", include_self=True)
    upper.scatter_reduce_(0, index, pieces[:, 1], reduce="amax", include_self=True)
    return torch.stack((lower, upper), 1)


def _merge_induced(pieces, piece_cells, rows, count, meets_losing):
    """Pieces whose hull meets ``L`` although none of the hull's pieces does."""
    hulls = _hulls(pieces, rows, count)
    hull_cells = torch.zeros(count, dtype=torch.long, device=pieces.device).scatter_(0, rows, piece_cells)
    hull_hit = meets_losing(hulls, hull_cells) & (torch.bincount(rows, minlength=count) > 1)
    if not bool(hull_hit.any()):
        return torch.zeros(len(pieces), dtype=torch.bool, device=pieces.device)
    candidates = hull_hit[rows]
    piece_hit = torch.zeros(count, dtype=torch.bool, device=pieces.device)
    piece_hit[rows[candidates][meets_losing(pieces[candidates], piece_cells[candidates])]] = True
    return (hull_hit & ~piece_hit)[rows]


def _transition_fails(problem, policy, boxes, cells, meets_losing):
    """Per policy-aligned box: SafeStep fails (uncertified tube or a successor meeting ``L``)."""
    fails = []
    for start in range(0, len(boxes), TRANSITION_BATCH):
        box, cell = boxes[start:start + TRANSITION_BATCH], cells[start:start + TRANSITION_BATCH]
        actions = policy.decode(policy.cell_values(cell))
        ok = tube_safe(problem, box, actions)
        pieces, piece_cells, parents, _, _ = next(successors(problem, policy, box, actions, ok))
        failed = ~ok
        failed[parents[meets_losing(pieces, piece_cells)]] = True
        fails.append(failed)
    return torch.cat(fails) if fails else torch.zeros(0, dtype=torch.bool, device=boxes.device)


def _certify_tubes_piecewise(problem, boxes, parents, actions, tube_ok, previous):
    """Re-certify failed tubes of source boxes inside merged rows on the rows' pieces.

    Trajectories start inside the pieces, not anywhere in their hull, so a
    source box is safe when the tube of its intersection with every piece is.
    """
    if previous is None or not len(previous.merged_pieces):
        return tube_ok
    failing = torch.where(~tube_ok)[0]
    if not len(failing):
        return tube_ok
    device = boxes.device
    # orignial pieces before merging
    pieces = previous.merged_pieces.to(device=device, dtype=boxes.dtype)
    # id in the merged sucessor boxes/row of the previous step
    piece_rows = previous.merged_piece_rows.to(device)
    failing_rows = parents[failing]
    selected = torch.isin(piece_rows, failing_rows)
    if not bool(selected.any()):
        return tube_ok
    pieces, owners = pieces[selected], piece_rows[selected]
    # Pair every piece with each failing source box of its row.
    order = torch.argsort(failing_rows)
    sorted_rows = failing_rows[order]
    starts = torch.searchsorted(sorted_rows, owners, right=False)
    counts = torch.searchsorted(sorted_rows, owners, right=True) - starts
    piece_ids = torch.repeat_interleave(torch.arange(len(pieces), device=device), counts)
    offsets = torch.arange(int(counts.sum()), device=device) - torch.repeat_interleave(counts.cumsum(0) - counts,
                                                                                         counts)
    sources = failing[order[starts[piece_ids] + offsets]]
    # the maximum/minimum is necessary due to goal removal, boundary faces also considered
    # we clip to the boundary of the pices and the merged source (due to goal removal)
    clipped = torch.stack((torch.maximum(pieces[piece_ids, 0], boxes[sources, 0]),
                           torch.minimum(pieces[piece_ids, 1], boxes[sources, 1])), 1)
    valid = (clipped[:, 0] <= clipped[:, 1]).all(1)
    clipped, sources = clipped[valid], sources[valid]
    if not len(clipped):
        return tube_ok
    safe = tube_safe(problem, clipped, actions[sources])
    total = torch.zeros(len(tube_ok), dtype=torch.long, device=device)
    certified = torch.zeros_like(total)
    total.index_add_(0, sources, torch.ones_like(sources))
    certified.index_add_(0, sources, safe.long())
    return tube_ok | ((total > 0) & (certified == total))


def detect_fixed_point(previous_boxes, previous_cells, boxes, cells):
    """Sufficient test for ``X_{k+1}`` contained in ``X_k`` (Algorithm 1, line 12).

    Each box of ``X_{k+1}`` must lie in the box of ``X_k`` in its cell; cells
    holding several boxes in ``X_k`` are not used, since the hull of those
    boxes could cover unreachable gaps.
    """
    if not len(boxes):
        return True
    if not len(previous_boxes):
        return False
    sorted_cells, order = torch.sort(previous_cells.to(previous_boxes.device))
    unique, counts = torch.unique_consecutive(sorted_cells, return_counts=True)
    starts = counts.cumsum(0) - counts
    keys = cells.to(previous_boxes.device)
    index = torch.searchsorted(unique, keys).clamp(max=len(unique) - 1)
    previous = previous_boxes[order[starts[index]]]
    covered = (unique[index] == keys) & (counts[index] == 1)
    covered &= (boxes[:, 0] >= previous[:, 0]).all(1) & (boxes[:, 1] <= previous[:, 1]).all(1)
    return bool(covered.all())
