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

    ``G`` is the obligation's game DAG, as tuples over the depths ``d``:
    ``cells[d]`` ``(N_d,)`` the policy cells of its nodes, ``values[d]``
    ``(N_d, K_d, A)`` the native values of their action slots, ``winning[d]``
    ``(N_d, K_d)`` the winning slots, and ``branches[d]`` ``(N_d,)`` the branch
    leading to each node, ``parent * K_{d-1} + slot`` (empty at depth 0).
    Anchored constraints (root at time 0) can never be backtracked.
    """

    obligation: Obligation
    cells: tuple
    values: tuple
    winning: tuple
    branches: tuple

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
    cells, games, one_step = {}, [], []

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
            one_step.append(constraint)
        else:
            games.append(constraint)
    # now process constraints to cell constraints
    for constraint in one_step:
        for cell_constraint in _one_step_cells(constraint):
            add(cell_constraint)
    # retunr games and 1-step compressed Cell constraints
    return tuple(games) + tuple(cells[key] for key in sorted(cells))


def _one_step_cells(constraint):
    """Cell constraints of a one-step repair constraint, one per cell of its root boxes.

    A value survives if some slot holding it wins at every root box of the
    cell; the values keep the slot order of the cell's first root box.
    """
    order, allowed = {}, {}
    for cell, values, winning in zip(constraint.cells[0].tolist(), constraint.values[0].tolist(),
                                     constraint.winning[0].tolist()):
        values = [tuple(int(v) for v in value) for value in values]
        won = {value for value, flag in zip(values, winning) if flag}
        if cell in allowed:
            allowed[cell] &= won
        else:
            order[cell], allowed[cell] = list(dict.fromkeys(values)), won
    if not order:
        raise ValueError("a one-step repair constraint has no root box")
    for cell in sorted(order):
        yield CellConstraint(cell, tuple(v for v in order[cell] if v in allowed[cell]), constraint.anchored,
                             (constraint,))
