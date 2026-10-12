"""Joint: one controller satisfying all repair constraints (CP-SAT, Section 2.2).

Each cell touched by a current constraint becomes a variable over the values
its games proposed there and its present value; every other cell keeps its
present value. A cell constraint requires one of its values. A game
constraint requires a winning value at every node it reaches: its roots are
always reached, and a node is reached when its parent is reached and the
parent's cell takes the value that leads to it. The objective is the summed
L1 distance of the native values to the original controller.

Every constraint is enabled by an assumption, so an infeasible problem names
a conflicting core. The current constraints in the core (after time 0) are
returned to move back; if there are none, the cells of the core's committed
constraints are released and the problem is solved again.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter

from ortools.sat.python import cp_model

from reacheco.constraints import CellConstraint, intersect

TIME_LIMIT = 60.0


@dataclass(frozen=True)
class JointResult:
    """``status`` is CP-SAT's; on success ``policy`` is the new controller.

    On a conflict, ``backtrack`` holds the obligations of the current
    constraints in the core that can move back (``replaced`` their
    identities), ``core_committed`` the committed constraints in the core and
    ``core_current`` the number of current ones.
    """

    status: str
    policy: object
    backtrack: tuple = ()
    replaced: frozenset = frozenset()
    core_committed: tuple = ()
    core_current: int = 0
    seconds: float = 0.0

    @property
    def satisfiable(self):
        return self.status in ("OPTIMAL", "FEASIBLE")


@dataclass(frozen=True)
class _Support:
    """A constraint and the cells it touches; a cell constraint merged from several keeps them as ``members``."""

    constraint: object
    cells: frozenset
    members: tuple = ()


def joint(policy, original, committed, current):
    """Solve for a controller meeting the ``committed`` and ``current`` constraints.

    ``policy`` is the controller being repaired, ``original`` the one whose
    values the objective stays close to.
    """
    started = perf_counter()
    committed_supports = tuple(map(_support, committed))
    current_supports = tuple(map(_support, current))
    released = frozenset()
    while True:
        problem = _Problem(policy, original, committed_supports, current_supports, released)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = TIME_LIMIT
        solver.parameters.num_workers = 16 #max(1, min(16, len(os.sched_getaffinity(0))))
        status = solver.solve(problem.model)
        name = solver.status_name(status)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return JointResult(name, problem.realize(solver), seconds=perf_counter() - started)
        if status != cp_model.INFEASIBLE:
            return JointResult(name, policy, seconds=perf_counter() - started)
        core = [problem.assumptions[int(index)] for index in solver.sufficient_assumptions_for_infeasibility()
                if int(index) in problem.assumptions]
        in_core = {id(s) for support in core for s in (support, *support.members)}
        conflicts = {}
        for support in current_supports:
            if id(support) in in_core:
                constraint = support.constraint
                for source in constraint.sources if isinstance(constraint, CellConstraint) else (constraint,):
                    if source.obligation.time > 0:
                        conflicts[source.constraint_id] = source
        if not conflicts:
            # A changed shared cell may enable a committed game's other branches:
            # their cells must be free before a conflict counts as final.
            cells = frozenset().union(*(support.cells for support in core)) - problem.free
            if cells:
                released |= cells
                continue
        current_ids = {id(support.constraint) for support in current_supports}
        core_committed = {}
        for support in core:
            for member in support.members or (support,):
                if id(member.constraint) not in current_ids:
                    core_committed.setdefault(id(member.constraint), member.constraint)
        return JointResult(name, policy, tuple(c.obligation for c in conflicts.values()), frozenset(conflicts),
                           tuple(core_committed.values()), sum(id(s) in in_core for s in current_supports),
                           perf_counter() - started)


def _support(constraint):
    if isinstance(constraint, CellConstraint):
        return _Support(constraint, frozenset({constraint.cell}))
    return _Support(constraint, frozenset(cell for cells in constraint.cells for cell in cells.tolist()))


def _key(value):
    return tuple(int(v) for v in value)

# keeping encodings alive between calls could speed up the algorithm quiet a bit.
class _Problem:
    """The CP-SAT model: one exactly-one choice per free cell, one assumption per constraint."""

    def __init__(self, policy, original, committed, current, released):
        self.policy = policy
        self.model = cp_model.CpModel()
        # the policy cells we allow repairs over, this is resticted to keep old changes
        # and make solving times shorter
        # released adds more cells to stay complete if infeasible previously
        self.free = frozenset(cell for support in current for cell in support.cells) | released
        supports = committed + current
        relevant = sorted(frozenset().union(*(s.cells for s in supports)))
        # values of actions in policy
        present = dict(zip(relevant, map(_key, policy.cell_values(relevant).tolist()))) if relevant else {}
        # originial policy actions
        reference = dict(zip(relevant, map(_key, original.cell_values(relevant).tolist()))) if relevant else {}
        # assign each cell its potential new values discovered by games
        candidates = _candidates(supports, self.free)
        self.literals, self.values, self.present = {}, {}, {}
        for cell in relevant:
            values = list(dict.fromkeys((*candidates.get(cell, ()), present[cell])))
            if cell in self.free:
                # encode that each free cell must pick one of the action values
                self.literals[cell] = tuple(self.model.new_bool_var("") for _ in values)
                self.model.add_exactly_one(self.literals[cell])
                self.values[cell], self.present[cell] = tuple(values), values.index(present[cell])
            else:
                # not modified cells
                self.literals[cell], self.values[cell], self.present[cell] = (True,), (present[cell],), 0
        self.index = {cell: {value: i for i, value in enumerate(values)} for cell, values in self.values.items()}
        # encode the L1 cost, the L1 cost is on the INTEGER values not on the final output actions
        terms = [sum(abs(a - b) for a, b in zip(value, reference[cell])) * self.literals[cell][i]
                 for cell in sorted(self.free) for i, value in enumerate(self.values[cell])
                 if value != reference[cell]]
        if terms:
            self.model.minimize(sum(terms))
        # add a hint that the previous solve might have been good
        for cell in sorted(self.free):
            for i, literal in enumerate(self.literals[cell]):
                self.model.add_hint(literal, int(i == self.present[cell]))
        # Games first; cell constraints of one cell (committed and current) share one assumption.
        self.assumptions = {}
        merged = {}
        for support in supports:
            if not isinstance(support.constraint, CellConstraint):
                self._encode_game(support)
                continue
            cell = support.constraint.cell
            if cell not in merged:
                merged[cell] = _Support(support.constraint, support.cells, members=(support,))
            else:
                previous = merged[cell]
                merged[cell] = _Support(intersect(previous.constraint, support.constraint),
                                        previous.cells | support.cells, members=previous.members + (support,))
        for support in merged.values():
            self._encode_cell(support)

    def realize(self, solver):
        """The controller with the chosen value in every free cell."""
        result = self.policy.copy()
        changed = [(cell, self.values[cell][index]) for cell in sorted(self.free)
                   for index in [next(i for i, literal in enumerate(self.literals[cell]) if solver.value(literal))]
                   if index != self.present[cell]]
        if changed:
            result.set_cell_values([cell for cell, _ in changed], [value for _, value in changed])
        return result

    def _choices(self, cell, values):
        index = self.index[cell]
        return sorted({index[v] for v in map(_key, values) if v in index})

    def _assume(self, support):
        enabled = self.model.new_bool_var("")
        self.model.add_assumptions([enabled])
        self.assumptions[enabled.index] = support
        return enabled

    def _encode_cell(self, support):
        constraint = support.constraint
        allowed = self._choices(constraint.cell, constraint.values)
        if len(self.literals[constraint.cell]) == 1 and allowed:
            return          # a fixed cell whose value is allowed
        self._require(self._assume(support), constraint.cell, allowed)

    def _encode_game(self, support):
        """Every reached node needs a winning value; a non-winning parent value reaches nothing."""
        constraint = support.constraint
        # a new literal per constraint
        enabled = self._assume(support)
        active, choices = {}, {}
        for depth, (cells, values, winning) in enumerate(zip(constraint.cells, constraint.values, constraint.winning)):
            cells, values, winning = cells.tolist(), values.tolist(), winning.tolist()
            if depth:
                # tried actions at each node
                slots = constraint.values[depth - 1].shape[1]
                branches = constraint.branches[depth].tolist()
            for node, cell in enumerate(cells):
                if depth == 0:
                    reached = enabled
                else:
                    parent, slot = divmod(branches[node], slots)
                    if (depth - 1, parent) not in active:
                        continue
                    index = self.index[parent_cells[parent]].get(_key(parent_values[parent][slot]))
                    if index not in choices[(depth - 1, parent)]:
                        continue        # a non-winning parent value never reaches this node
                    reached = self._reached(active[(depth - 1, parent)], parent_cells[parent], index)
                active[(depth, node)] = reached
                choices[(depth, node)] = self._choices(cell, [v for v, flag in zip(values[node], winning[node]) if flag])
                self._require(reached, cell, choices[(depth, node)])
            parent_cells, parent_values = cells, values

    def _require(self, active, cell, allowed):
        """``active`` implies that the cell takes one of the ``allowed`` values."""
        literals = self.literals[cell]
        if len(literals) == 1:
            if 0 not in allowed:
                self.model.add_bool_or([active.Not()])
            return
        if len(allowed) == len(literals):
            return
        self.model.add_bool_or([active.Not(), *(literals[i] for i in allowed)])

    def _reached(self, parent, cell, index):
        """A literal for: the ``parent`` node is reached and its cell takes value ``index``."""
        literals = self.literals[cell]
        if len(literals) == 1:
            return parent       # a cell with a single value always takes it
        reached = self.model.new_bool_var("")
        self.model.add_bool_or([reached.Not(), parent])
        self.model.add_bool_or([reached.Not(), literals[index]])
        self.model.add_bool_or([parent.Not(), literals[index].Not(), reached])
        return reached


def _candidates(supports, free):
    """Per free cell, the values the constraints offer there (all game slots), in order of appearance."""
    candidates = {}
    for support in supports:
        constraint = support.constraint
        if isinstance(constraint, CellConstraint):
            if constraint.cell in free:
                candidates.setdefault(constraint.cell, {}).update(dict.fromkeys(constraint.values))
            continue
        for cells, values in zip(constraint.cells, constraint.values):
            for cell, node_values in zip(cells.tolist(), values.tolist()):
                if cell in free:
                    bucket = candidates.setdefault(cell, {})
                    for value in map(_key, node_values):
                        bucket.setdefault(value)
    return {cell: tuple(values) for cell, values in candidates.items()}
