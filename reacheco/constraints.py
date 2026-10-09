"""Repair constraints and their per-cell compression.

FindCont yields one repair constraint ``gamma = (G, i)`` per won obligation.
A constraint of an obligation with horizon 1 only restricts the cells of its
root boxes: each root box needs one of its winning values. Since the
controller has one value per cell, the constraints with roots in one cell
reduce to the intersection of their winning values there (a ``CellConstraint``).
Deeper constraints stay games and are encoded node by node by Joint.

Constraints rooted at time 0 (anchored) are kept apart from the others, so
the others can still be dropped or moved back in a conflict.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CellConstraint:
    """The cell's native value must be one of ``values`` (in their order of first appearance).

    ``sources`` are the one-step repair constraints compressed into it; they
    are kept for current constraints, so a conflict can move them back.
    """

    cell: int
    values: tuple
    anchored: bool
    sources: tuple = ()

    @property
    def constraint_id(self):
        return ("cell", self.cell, self.anchored)


def constraint_id(constraint):
    """Identity of a constraint across repairs: its cell, or its obligation."""
    if isinstance(constraint, CellConstraint):
        return constraint.constraint_id
    obligation = constraint.obligation
    return ("game", obligation.root, obligation.failure_time, obligation.covers)


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
    cells, games, groups = {}, [], {}

    def add(constraint):
        if not keep_sources and constraint.sources:
            constraint = replace(constraint, sources=())
        key = constraint.constraint_id
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
    for (_, _, anchored), group in groups.items():
        for constraint in _one_step_cells(group, anchored):
            add(constraint)
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
