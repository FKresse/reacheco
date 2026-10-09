"""Obligations, repair constraints and the per-cell compression of one-step constraints.

An obligation ``(B, h)`` asks for a continuation from box ``B`` of ``X_t``
that keeps ``SafeStep`` until the failure step ``t + h``. FindCont yields one
repair constraint ``gamma = (G, i)`` per won obligation. A constraint of an
obligation with horizon 1 only restricts the cells of its root boxes: each
root box needs one of its winning values. Since the controller has one value
per cell, the constraints with roots in one cell reduce to the intersection
of their winning values there (a ``CellConstraint``). Deeper constraints stay
games and are encoded node by node by Joint.

Constraints rooted at time 0 (anchored) are kept apart from the others, so
the others can still be dropped or moved back in a conflict.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch


@dataclass(frozen=True, order=True)
class BoxRef:
    """A box of the frontier ``X_time`` (``index`` may exceed the stored rows for replayed boxes)."""

    time: int
    index: int


@dataclass(frozen=True)
class Obligation:
    """``(B, h)``: find a continuation from ``box`` that is safe until ``failure_time``.

    ``root`` refers to the forward history of the ``repair`` (the index of the
    BRepair call) the obligation belongs to, so ``(repair, root,
    failure_time)`` identifies it. ``cell`` is the policy cell of ``box``, or
    ``-1`` if it is not known yet.
    """

    root: BoxRef
    failure_time: int
    box: torch.Tensor = field(compare=False, repr=False)
    cell: int = field(default=-1, compare=False)
    repair: int = 0

    @property
    def time(self):
        return self.root.time

    @property
    def horizon(self):
        return self.failure_time - self.root.time


@dataclass(frozen=True)
class RepairConstraint:
    """``gamma = (G, i)``: the winning continuation found for one obligation.

    ``game`` holds the DAG (a ``continuation.Game``, shared with the batch),
    ``winning`` the winning action slots per depth, ``index`` the obligation
    within the batch. Anchored constraints (root at time 0) can never be
    backtracked.
    """

    game: object
    winning: tuple
    index: int

    @property
    def obligation(self):
        return self.game.obligations[self.index]

    @property
    def root_time(self):
        return self.obligation.time

    @property
    def anchored(self):
        return self.root_time == 0

    @property
    def constraint_id(self):
        obligation = self.obligation
        return ("game", obligation.repair, obligation.root, obligation.failure_time)


@dataclass(frozen=True)
class CellConstraint:
    """
    A CellConstraint is a optimization for one-step constraints, it specifically helps 
    in docking2d.
    Basically, the intersection of allowed values is computed and obligations from the same 
    cell are compressed.
    The cell's native value must be one of ``values`` (in their order of first appearance).
    ``sources`` are the one-step repair constraints compressed into it; they
    are kept for current constraints, so a conflict can move them back.
    """

    cell: int
    values: tuple # list of allowed actions (intersection of sources)
    anchored: bool
    sources: tuple = ()

    @property
    def constraint_id(self):
        return ("cell", self.cell, self.anchored)


def intersect(first, second, *, keep_sources=True):
    """Both constraints of one cell together: the values allowed by both, in ``first``'s order."""
    if first.cell != second.cell:
        raise ValueError("only constraints of one cell intersect")
    allowed = set(second.values)
    return CellConstraint(first.cell, tuple(v for v in first.values if v in allowed), first.anchored,
                          first.sources + second.sources if keep_sources else ())


def compress(constraints, *, keep_sources=True):
    """Replace the one-step repair constraints by cell constraints and merge cell constraints per cell.

    Returns the deeper repair constraints (in their order), then the cell
    constraints sorted by identity.
    """
    # anchored and non-anchored are always merged into different keys
    cells, games, groups = {}, [], {}

    def add(constraint):
        if not keep_sources and constraint.sources:
            constraint = replace(constraint, sources=())
        key = constraint.constraint_id # this keep anchored and non-anchored seperate
        cells[key] = constraint if key not in cells else intersect(cells[key], constraint,
                                                                   keep_sources=keep_sources)

    for constraint in constraints:
        if isinstance(constraint, CellConstraint):
            add(constraint)
        elif constraint.obligation.horizon == 1:
            groups.setdefault((id(constraint.game), id(constraint.winning[0]), constraint.anchored),
                              []).append(constraint)
        else:
            games.append(constraint)
    # now process constraints to cell constraints
    for (_, _, anchored), group in groups.items():
        for constraint in _one_step_cells(group, anchored):
            add(constraint)
    # retunr games and 1-step compressed Cell constraints
    return tuple(games) + tuple(cells[key] for key in sorted(cells))


def _one_step_cells(group, anchored):
    """Cell constraints of the one-step repair constraints of one game, by cell (ascending)."""
    game, winning = group[0].game, group[0].winning[0]
    by_index = {}
    for constraint in group:
        by_index.setdefault(constraint.index, []).append(constraint)
    layer, transition = game.layers[0], game.transitions[0]
    rows, sources = {}, {}
    for row, (owner, cell) in enumerate(zip(layer.owners.tolist(), layer.cells.tolist())):
        if owner in by_index:
            rows.setdefault(cell, []).append(row)
            sources.setdefault(cell, {})[owner] = None
    if set().union(*map(set, sources.values())) != set(by_index):
        raise ValueError("a one-step repair constraint has no root box")
    for cell in sorted(rows):
        # A value survives if some slot holding it wins at every root box of the cell.
        allowed = None
        order = []
        for row in rows[cell]:
            values = [tuple(int(v) for v in value) for value in transition.values[row].tolist()]
            won = {value for value, flag in zip(values, winning[row].tolist()) if flag}
            if allowed is None:
                order = list(dict.fromkeys(values))
                allowed = won
            else:
                allowed &= won
        yield CellConstraint(cell, tuple(v for v in order if v in allowed), anchored,
                             tuple(c for index in sources[cell] for c in by_index[index]))
