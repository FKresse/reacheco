"""Backtrack: move unresolved obligations to the parents of their boxes (Section 2.2).

An obligation ``(B, h)`` without a safe continuation moves to the box ``B'``
of ``X_{t-1}`` it came from, as ``(B', h + 1)``, keeping its failure time.
Merged rows have several parents, so a box of a merged row is first
unmerged: its sources one step earlier are replayed, without merging, up to
the failure time (``reach_step(merge=False)``, against the losing set of the
forward pass). Only the replayed boxes that fail again lead to obligations,
and each such chain of single-parent boxes is kept, so later steps back walk
it without replaying. A failure whose replay fails nowhere was created by
merging alone and needs no repair.

Replays are lazy: each step back unmerges one more step of the history. The
cells of the backward ancestor cone of the failure (in the merged history)
are protected from merging in later forward passes (``merge_disabled_at``),
so the unmerged failing chains are reproduced there.
"""
from __future__ import annotations

import torch

from reacheco.constraints import BoxRef, Obligation
from reacheco.reach import reach_step


class Backtrack:
    """Backtracking for one failure of a forward pass.

    ``frontiers`` ``X_0 .. X_T`` and ``records`` (``X_t -> X_{t+1}``) come from
    a forward pass against ``losing_set`` that failed at step ``T``;
    ``merge_disabled`` and ``merge_disabled_at`` are its merge masks.
    ``merge_disabled_at`` grows by the protections of every replay. ``repair``
    is the index of the repair (in the run) the obligations belong to.
    """

    def __init__(self, problem, policy, frontiers, records, losing_set, *, merge_disabled=(),
                 merge_disabled_at=None, repair=0):
        if len(frontiers) != len(records) + 1 or not bool(records[-1].failing.any()):
            raise ValueError("backtracking needs the frontiers and records of a failed forward pass")
        self.problem, self.policy = problem, policy
        self.frontiers, self.records = frontiers, records
        self.failure_time = len(records)
        self.repair = repair
        # The replays must fail exactly as the forward pass did, so they use L as it was then.
        self._meets_losing = losing_set.clone().intersects
        self.merge_disabled = set(merge_disabled)
        self.merge_disabled_at = {int(t): set(cells) for t, cells in (merge_disabled_at or {}).items()}
        failing = records[-1].failing
        self._failure_cells = set(records[-1].cells[failing].tolist())
        self._cone = _ancestor_cone(records, torch.where(failing)[0])
        self._cone_cells = {}
        self._parent = {}         # replayed box -> its unique parent
        self._nodes = {}          # replayed box -> (box, cell)
        self._parent_cache = {}
        self._next_index = [len(boxes) for boxes in frontiers]

    def failures(self):
        """The one-step obligations at the parents of the failing boxes (Algorithm 2's ``F``).

        Empty if the failure was created by merging alone.
        """
        refs = [BoxRef(self.failure_time, row) for row in torch.where(self.records[-1].failing)[0].tolist()]
        return self([Obligation(ref, self.failure_time, *self._node(ref), repair=self.repair) for ref in refs])

    def __call__(self, obligations):
        """``Backtrack(F_o)``: the obligations one step earlier, one per parent box."""
        obligations = tuple(dict.fromkeys(obligations))
        if not obligations:
            return ()
        if any(o.failure_time != self.failure_time or o.repair != self.repair or o.time == 0 for o in obligations):
            raise ValueError("only obligations of this failure after time 0 can move back")
        parents = self._parents_many(o.root for o in obligations)
        merged_parents = self._parents_many(p for ps in parents.values() for p in ps if self._merged(p))
        grouped, requests = {}, {}
        for obligation in obligations:
            candidates = parents[obligation.root]
            if obligation.root in self._parent and not self._merged(candidates[0]):
                # Already on a replayed failing chain: walk it back without replaying.
                grouped.setdefault(candidates[0], None)
                continue
            # To unmerge a merged parent at t, replay from its sources at t - 1.
            for parent in candidates:
                for seed in merged_parents[parent] if self._merged(parent) else (parent,):
                    requests.setdefault(seed, {})[obligation] = None
        if requests:
            # replay the requested backtracks forward wihtout merging
            self._recover(requests, grouped)
        return tuple(Obligation(root, self.failure_time, *self._node(root), repair=self.repair) for root in grouped)

    # ------------------------------------------------------------ history

    def _node(self, ref):
        """``(box, cell)`` of a box of the history (cell ``-1`` at time 0)."""
        if ref in self._nodes:
            return self._nodes[ref]
        cell = int(self.records[ref.time - 1].cells[ref.index]) if ref.time else -1
        return self.frontiers[ref.time][ref.index], cell

    def _merged(self, ref):
        # -1 in row_sources is merged
        return (0 < ref.time and ref.index < len(self.frontiers[ref.time])
                and int(self.records[ref.time - 1].row_sources[ref.index]) < 0)

    def _parents_many(self, refs):
        """Parents of boxes: the replayed parent, the unique source, or all sources of a merged row."""
        result, pending = {}, {}
        # deduplicate
        for ref in dict.fromkeys(refs):
            if ref in self._parent:
                result[ref] = (self._parent[ref],)
            elif ref in self._parent_cache:
                result[ref] = self._parent_cache[ref]
            else:
                # we actualyy have to look for the transition the produced this
                record = self.records[ref.time - 1]
                source = int(record.row_sources[ref.index])
                if source >= 0:
                    result[ref] = (BoxRef(ref.time - 1, int(record.source_parents[source])),)
                else:
                    # merged box do it below
                    pending.setdefault(ref.time, []).append(ref)
        for time, refs in pending.items():
            # look at each timestep indivdually
            record = self.records[time - 1]
            requested = torch.zeros(len(record.cells), dtype=torch.bool)
            requested[[ref.index for ref in refs]] = True
            selected = requested[record.piece_rows]
            pairs = torch.stack((record.piece_rows[selected],
                                 record.source_parents[record.piece_sources[selected]]), 1).unique(dim=0)
            sources = {ref.index: [] for ref in refs}
            for child, parent in pairs.tolist():
                sources[child].append(BoxRef(time - 1, parent))
            for ref in refs:
                result[ref] = self._parent_cache[ref] = tuple(sources[ref.index])
        return result

    # ------------------------------------------------------------- replay

    def _recover(self, requests, grouped):
        """Replay the seeds and add the ancestor of each failing replayed box to ``grouped``."""
        seeds = tuple(requests)
        start = min(seed.time for seed in seeds)
        history, records, entries, owners, failing = self._replay(seeds, start)
        if not bool(failing.any()):
            return
        parents, nodes = _ancestry(history, records, torch.where(failing)[0], start, entries)
        references = self._store(parents, nodes, records, start, entries)
        # walk backwards
        for row in torch.where(failing)[0].tolist():
            leaf = BoxRef(self.failure_time, row)
            for obligation in requests[seeds[int(owners[row])]]:
                root = leaf
                while root.time >= obligation.time:
                    root = parents[root]
                grouped.setdefault(references[root], None)

    def _replay(self, seeds, start):
        """Replay the seeds from their times to the failure, merging only in the last step."""
        times = {}
        for owner, seed in enumerate(seeds):
            times.setdefault(seed.time, []).append((owner, seed))
        # Only intermediate replay steps use the cone protections; the last uses the failure cells.
        # cone protection disables merging in the backwards ancestor of the failure. 
        # this is an important optimization as cells outside the cone can be remerged
        # enabling fast forwards that dont blow up.
        # protection is done later, computes cone here.
        while start + 1 < self.failure_time and start + 1 not in self._cone_cells:
            time, cells = next(self._cone)
            self._cone_cells[time] = cells
        policy = self.policy
        # initalize an empty frontier
        frontier = policy.lower.new_empty((0, 2, len(policy.thresholds)))
        owners = torch.empty(0, dtype=torch.long)
        history, records, entries = [], [], {}
        for time in range(start, self.failure_time):
            # injected gets the index in the frontier (seed) of the obligation and the index of the owner
            injected = times.get(time, ())
            # inject obligations at their associated timestamp
            for offset, (_, seed) in enumerate(injected):
                entries[BoxRef(time, len(frontier) + offset)] = seed
            if injected:
                boxes = torch.stack([self._node(seed)[0] for _, seed in injected]).to(frontier)
                frontier = torch.cat((frontier, boxes))
                owners = torch.cat((owners, torch.tensor([owner for owner, _ in injected])))
            history.append(frontier.cpu())
            final = time + 1 == self.failure_time
            disabled = (self.merge_disabled | self.merge_disabled_at.get(time + 1, set()) | self._failure_cells
                        if final else ())
            frontier, record = reach_step(frontier, policy, self.problem, self._meets_losing, merge=final,
                                          merge_disabled=disabled, lookahead=False)
            records.append(record)
            unique = record.row_sources >= 0
            following = torch.full((len(record.cells),), -1, dtype=torch.long)
            following[unique] = owners[record.source_parents[record.row_sources[unique]]]
            owners = following
            # apply cone protection, no need in first timestep
            protected = self._failure_cells if final else self._cone_cells.get(time + 1, frozenset())
            self.merge_disabled_at.setdefault(time + 1, set()).update(protected)
            failing = record.failing
            if bool(failing.any()) and not (final and bool(unique[failing].all())):
                raise RuntimeError(f"the replay failed at step {time + 1} or in a merged box, "
                                   f"unlike the forward pass (failure at step {self.failure_time})")
        history.append(frontier.cpu())
        return history, records, entries, owners, failing

    def _store(self, parents, nodes, records, start, entries):
        """Give the replayed boxes stable references (after the rows of the history) and keep their chains."""
        references = {}
        for ref, box in nodes.items():
            if ref in entries:
                references[ref] = entries[ref]
                continue
            stored = BoxRef(ref.time, self._next_index[ref.time])
            self._next_index[ref.time] += 1
            references[ref] = stored
            self._nodes[stored] = (box, int(records[ref.time - start - 1].cells[ref.index]))
        for child, parent in parents.items():
            self._parent[references[child]] = references[parent]
        return references


def _ancestry(history, records, failing, start, entries):
    """Unique-parent chains from the failing replayed boxes back to the injected seeds.

    Returns the parent of every visited box and the visited boxes, leaves
    first, then level by level.
    """
    failure_time = start + len(records)
    level = tuple(BoxRef(failure_time, row) for row in failing.tolist())
    nodes = {ref: history[-1][ref.index] for ref in level}
    parents = {}
    while level and level[0].time > start:
        record = records[level[0].time - 1 - start]
        following = {}
        for child in level:
            if child in entries:
                continue
            source = int(record.row_sources[child.index])
            parent = BoxRef(child.time - 1, int(record.source_parents[source]))
            parents[child] = parent
            nodes.setdefault(parent, history[parent.time - start][parent.index])
            following[parent] = None
        level = tuple(following)
    return parents, nodes


def _ancestor_cone(records, failing):
    """Cells of the backward ancestor cone of the failing rows, ``(time, cells)`` from the failure back to 0.

    The cells at time ``t`` are those of the cone's rows of ``X_t`` and of the
    source boxes they were split into.
    """
    rows, cells = failing, {len(records): set()}
    for step in range(len(records) - 1, -1, -1):
        record = records[step]
        cells.setdefault(step + 1, set()).update(record.cells[rows].tolist())
        yield step + 1, frozenset(cells.pop(step + 1))
        edges = torch.isin(record.piece_rows, rows)
        sources = record.piece_sources[edges]
        cells.setdefault(step, set()).update(record.source_cells[sources].tolist())
        rows = torch.unique(record.source_parents[sources])
    yield 0, frozenset(cells[0])
