"""ReachECO (Algorithm 1) and BRepair (Algorithm 2).

The forward pass runs until it fails; BRepair then searches continuations for
the failing obligations (FindCont), moves the unresolved ones back
(Backtrack), and realizes all constraints in one controller (Joint). The
forward pass resumes at the earliest step the repair can affect. When the
pass is certified, changed cells that the pass never read are reverted
(pruning), so verifying the result again reproduces exactly this pass.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import torch

from reacheco.backtrack import Backtrack
from reacheco.constraints import compress
from reacheco.continuation import find_continuations
from reacheco.joint import joint
from reacheco.losing import LosingSet
from reacheco.verify import verify


@dataclass
class BRepairResult:
    """``status``: ``"repaired"``, ``"losing-set-expanded"`` or ``"unknown"`` (``reason`` says why).

    ``current`` are the compressed current constraints of a repair,
    ``committed`` the committed constraints that remain (conflicting ones
    after time 0 may be dropped), ``merge_disabled_at`` the merge protections
    after backtracking.
    """

    status: str
    policy: object
    current: tuple
    committed: tuple
    merge_disabled_at: dict
    losing_delta: torch.Tensor
    reason: str | None = None
    statistics: Counter = field(default_factory=Counter)

    @property
    def losing_added(self):
        return len(self.losing_delta)


def brepair(problem, policy, original, frontiers, records, losing_set, backend, committed, *, learn=True,
            merge_disabled=(), merge_disabled_at=None, index=0):
    """Algorithm 2 for the failure at the end of ``frontiers``; ``losing_set`` grows in place."""
    statistics = Counter()
    backtrack = Backtrack(problem, policy, frontiers, records, losing_set, merge_disabled=merge_disabled,
                          merge_disabled_at=merge_disabled_at, repair=index)
    added = []

    def result(status, repaired=policy, current=(), reason=None):
        delta = torch.cat(added) if added else torch.empty((0, 2, len(policy.thresholds)))
        return BRepairResult(status, repaired, current, committed, backtrack.merge_disabled_at, delta, reason,
                             statistics)

    unresolved = backtrack.failures()
    if not unresolved:
        # Every failing box has passing parents: merging alone created the failure.
        return result("repaired")
    committed, resolved = tuple(committed), {}
    while True:
        # backtracking until all obligations are resolved (lines 4,5 and 15-18 in algorithm 2)
        while unresolved:
            found = find_continuations(unresolved, backend, problem, policy, losing_set, learn=learn)
            added.append(found.losing_delta)
            statistics.update({key: value for key, value in found.statistics.items()})
            statistics["findcont_calls"] += 1
            statistics.update(f"horizon_{o.horizon}" for o in unresolved)      # games per depth
            for constraint in found.constraints:
                resolved[constraint.constraint_id] = constraint
            if not found.unresolved:
                break
            if any(o.time == 0 for o in found.unresolved):
                return result("unknown", reason="initial-box-losing")
            unresolved = backtrack(found.unresolved)
        unresolved = ()
        # compress one-step constraints to cell constraints, an optimization for Docking2d
        # which has to deal with many 1-step games. 
        committed = compress(committed, keep_sources=False)
        current = compress(tuple(resolved.values()))
        solved = joint(policy, original, committed, current)
        statistics[f"joint_{solved.status.lower()}"] += 1
        statistics["joint_seconds"] += solved.seconds
        if solved.satisfiable:
            return result("repaired", solved.policy, compress(current, keep_sources=False))
        # CP-SAT's time limit (status UNKNOWN) leaves the conflict undecided.
        conflict = "no-backtrackable-obligation" if solved.status == "INFEASIBLE" else "joint-timeout"
        if solved.status == "INFEASIBLE" and not solved.backtrack and resolved:
            if not solved.core_current:
                raise RuntimeError("the joint constraints conflict without any current constraint")
            # Committed constraints after time 0 in the conflict were solved against an
            # older L or controller; drop them and let later forward passes re-derive them.
            dropped = {id(c) for c in solved.core_committed if not c.anchored}
            if dropped:
                committed = tuple(c for c in committed if id(c) not in dropped)
                statistics["committed_dropped"] += len(dropped)
                continue
            if solved.core_committed:
                conflict = "anchored-conflict"
        if not solved.backtrack:
            return result("losing-set-expanded" if sum(map(len, added)) else "unknown", reason=conflict)
        for key in solved.replaced:
            resolved.pop(key, None)
        unresolved = backtrack(solved.backtrack)


@dataclass
class RepairResult:
    """Outcome of ReachECO.

    ``status`` is ``"safe"`` (certified up to ``horizon``, ``inf`` for a fixed
    point or the goal) or ``"unknown"`` (``reason``). ``policy`` is the pruned
    controller, ``repaired`` the one before pruning, ``committed`` the
    committed constraints (for further runs on other initial sets). A forward
    pass with ``merge_disabled_at`` and ``losing_set`` reproduces the
    certifying pass.
    """

    status: str
    horizon: float | None
    policy: object
    repaired: object
    merge_disabled_at: dict
    losing_set: LosingSet
    committed: tuple
    repairs: list
    reason: str | None = None
    seconds: float = 0.0

    def save(self, directory):
        """``controller.pt``, ``merge_mask.json``, ``losing_set.json`` and ``report.json`` in ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.policy.save(directory / "controller.pt")
        mask = {"format": "reacheco-merge-mask", "version": 1,
                "at": {str(t): sorted(cells) for t, cells in sorted(self.merge_disabled_at.items()) if cells}}
        (directory / "merge_mask.json").write_text(json.dumps(mask) + "\n")
        (directory / "losing_set.json").write_text(json.dumps({"boxes": self.losing_set.boxes.tolist()}) + "\n")
        report = {"status": self.status, "horizon": self.horizon, "reason": self.reason, "seconds": self.seconds,
                  "overrides": len(self.policy.overrides), "pruned_cells": len(self.repaired.changed_cells(self.policy)),
                  "losing_boxes": len(self.losing_set), "repairs": self.repairs}
        (directory / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")


def load_merge_mask(path):
    """``merge_disabled_at`` of a saved mask (``{"at": {t: cells}}``)."""
    return {int(t): set(cells) for t, cells in json.loads(Path(path).read_text())["at"].items()}


def repair(problem, policy, backend, *, losing_set=None, learn=True, merge=True, prune=True, committed=(),
           greedy=False, on_repair=None):
    """Algorithm 1: verify, repair each failure, resume at the earliest affected step.

    ``policy`` is not modified. ``losing_set`` (default: just ``U``) grows when
    ``learn``. ``committed`` are committed constraints of earlier runs (on
    other initial sets) that every repair keeps. ``on_repair(entry)`` is called
    with the report of every repair.

    ``greedy`` resumes just before the deepest game of each repair instead
    (the prototype's greedy restart): faster on long horizons, but a ``"safe"``
    result is a certificate only if the run needed no repair that changed the
    controller; Docking2D repeats runs until one is clean.
    """
    if greedy and prune:
        raise ValueError("pruning needs one consistent certifying pass; the greedy restart does not give one")
    started = perf_counter()
    original, policy = policy, policy.copy()
    losing_set = LosingSet(problem, policy) if losing_set is None else losing_set
    committed = {c.constraint_id: c for c in committed}
    merge_disabled_at, repairs = {}, []
    frontiers = records = None
    status = None
    while True:
        # verify performs ReachStep until a fixed-point is reached (config depending),
        # a failure is found, or the budget T is reached.
        result = verify(problem, policy, losing_set.intersects, merge=merge, merge_disabled_at=merge_disabled_at,
                        frontiers=frontiers, records=records)
        frontiers, records = result.frontiers, result.records
        if result.certified:
            status, reason = "safe", None
            break
        repair_started = perf_counter()
        outcome = brepair(problem, policy, original, frontiers, records, losing_set, backend,
                          tuple(committed.values()), learn=learn, merge_disabled_at=merge_disabled_at,
                          index=len(repairs))
        # the losing set is already expanded inside brepair, this earlier expansion
        # makes it possible for later games to benefit from knowledge gained
        # it is listed in Algorithm 1 for clarity of the procedure.
        # merge protected cells
        protected = {t: cells - merge_disabled_at.get(t, set()) for t, cells in outcome.merge_disabled_at.items()}
        protected = {t: cells for t, cells in protected.items() if cells}
        merge_disabled_at = outcome.merge_disabled_at
        committed = {c.constraint_id: c for c in outcome.committed}
        changed = policy.changed_cells(outcome.policy)
        entry = {"failure_step": result.failure_step, "failing": int(result.failing.sum()),
                 "status": outcome.status, "reason": outcome.reason, "changed_cells": len(changed),
                 "losing_added": outcome.losing_added, "protected_cells": sum(map(len, protected.values())),
                 "seconds": perf_counter() - repair_started, **outcome.statistics}
        if outcome.status == "unknown" and not protected:
            status, reason = "unknown", outcome.reason
        elif not changed and not outcome.losing_added and not protected:
            status, reason = "unknown", "no-progress"
        if status == "unknown":
            repairs.append(entry)
            if on_repair:
                on_repair(entry)
            break
        if outcome.status == "repaired":
            # for committed constraints we no longer care about sources 
            # as they cannot be backtracked anyways.
            committed = {c.constraint_id: c
                         for c in compress((*committed.values(), *outcome.current), keep_sources=False)}
        policy = outcome.policy
        # Restart policy
        if greedy:
            # if using greedy we need at least one clean full pass without repairs.
            deepest = max((int(key[8:]) for key in outcome.statistics if key.startswith("horizon_")), default=1)
            restart = max(0, result.failure_step - deepest)
            if protected:
                # A newly protected row at t is formed by the step t - 1 -> t.
                restart = min(restart, max(0, min(protected) - 1))
        else:
            restart = earliest_affected(frontiers, records, changed, protected, outcome.losing_delta, losing_set)
        entry["restart"] = restart
        repairs.append(entry)
        if on_repair:
            on_repair(entry)
        frontiers, records = frontiers[:restart + 1], records[:restart]
    repaired = policy
    # in case we are done repairing, prune the patch to only 'touched' cells in the current controllers 
    # forward verification.
    if status == "safe" and prune:
        policy = _prune(original, repaired, records)
    return RepairResult(status, result.horizon if status == "safe" else None, policy, repaired,
                        merge_disabled_at, losing_set, tuple(committed.values()), repairs, reason,
                        perf_counter() - started)


def earliest_affected(frontiers, records, changed, protected, delta=(), losing_set=None):
    """EarliestAffected: the first step whose result can differ after a repair.

    A step is affected if it read a changed cell, produced a newly protected
    one, or queried ``L`` with a box meeting the boxes ``delta`` just added to
    ``L``: L is queried with the rows of ``X_{k+1}`` and their pieces in step
    ``k``, and with the pieces of step ``k + 1`` by the merge lookahead in step
    ``k``. A row contains its pieces, so checking the rows suffices. The
    frontiers before the affected step are unchanged, so the forward pass
    resumes there (at ``X_0`` if nothing is affected).
    """
    changed = torch.tensor(sorted(changed), dtype=torch.long)
    added = LosingSet(losing_set.problem, losing_set.policy, delta) if len(delta) else None

    def meets_added(step):
        device = losing_set.policy.device
        return bool(added.intersects(frontiers[step + 1].to(device), records[step].cells.to(device)).any())

    following = added is not None and meets_added(0)
    for step, record in enumerate(records):
        current = following
        following = added is not None and step + 1 < len(records) and meets_added(step + 1)
        if len(changed) and bool(torch.isin(record.touched_cells, changed).any()):
            return step
        cells = protected.get(step + 1)
        if cells and bool(torch.isin(record.cells, torch.tensor(sorted(cells), dtype=torch.long)).any()):
            return step
        if current or following:
            return step
    return 0


def _prune(original, repaired, records):
    """Keep only the repairs of cells the certifying pass read (including merge lookahead reads)."""
    touched = set(torch.cat([record.touched_cells for record in records]).tolist()) if records else set()
    kept = sorted(original.changed_cells(repaired) & touched)
    pruned = original.copy()
    if kept:
        pruned.set_cell_values(kept, repaired.cell_values(kept))
    return pruned
