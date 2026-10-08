"""FindCont: search safe continuations for obligations (Section 2.2).

An obligation ``(B, h)`` asks for a continuation from box ``B`` that keeps
``SafeStep`` for ``h`` steps. FindCont expands, for a batch of obligations,
the game DAG of the actions the backend ``W`` proposes, then solves it
backwards: an action wins at a node if its tube is certified and every
successor piece avoids ``L`` and wins in turn (or the horizon is reached).
Each obligation whose root wins yields a repair constraint ``gamma = (G, i)``:
its game together with the winning actions at every node. While solving,
nodes at which every action leads into ``L`` are certified losing and added
to ``L`` (Equation 11), together with their whole policy cell if every
action also fails from all of the cell. Losing is relative to ``W``.

All successor computations go through ``reach.successors``, so continuation
games and forward verification share one transition semantics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

import torch

from reacheco.boxes import cell_pieces, cut_goal, inside_any
from reacheco.reach import successors
from reacheco.tubes import tube_safe

BRANCH_BATCH = 4096
PIECE_BUDGET = 16384


# ------------------------------------------------------------ obligations

@dataclass(frozen=True, order=True)
class BoxRef:
    """A box of the frontier ``X_time`` (``index`` may exceed the stored rows for replayed boxes)."""

    time: int
    index: int


@dataclass(frozen=True)
class Obligation:
    """``(B, h)``: find a continuation from ``box`` that is safe until ``failure_time``.

    ``covers`` names the failures (or joint conflicts) the obligation answers
    for; with ``root`` and ``failure_time`` it identifies the obligation.
    ``cell`` is the policy cell of ``box``, or ``-1`` if it is not known yet.
    """

    root: BoxRef
    failure_time: int
    covers: frozenset
    box: torch.Tensor = field(compare=False, repr=False)
    cell: int = field(default=-1, compare=False)

    @property
    def time(self):
        return self.root.time

    @property
    def horizon(self):
        return self.failure_time - self.root.time


# ------------------------------------------------------------- game DAGs

@dataclass(frozen=True)
class GameLayer:
    """Policy-aligned nodes at one depth: ``boxes``, ``cells``, the obligation
    each node belongs to (``owners``), and whether it met ``L`` when expanded."""

    boxes: torch.Tensor
    cells: torch.Tensor
    owners: torch.Tensor
    losing: torch.Tensor


@dataclass(frozen=True)
class GameTransition:
    """The actions tried at the nodes of one layer and where they lead.

    ``values`` ``(N, K, A)`` are the native values of the ``K`` action slots
    per node. Per branch (node, slot), flattened as ``node * K + slot``:
    ``expanded``, ``terminal`` (horizon reached, tube uncertified, or all
    pieces in the goal), ``safe_tubes``, ``goal_reached``. Terminal branches
    keep no pieces; their pieces are summarized against ``L`` at expansion
    (``terminal_intersects``, ``terminal_contained``). Pieces of the other
    branches are ``post_boxes``/``post_cells`` with their ``post_branches``;
    ``successor_posts`` lists the pieces that became nodes of the next layer,
    ``successor_branches`` their branch.
    """

    values: torch.Tensor
    expanded: torch.Tensor
    terminal: torch.Tensor
    safe_tubes: torch.Tensor
    goal_reached: torch.Tensor
    terminal_intersects: torch.Tensor
    terminal_contained: torch.Tensor
    post_boxes: torch.Tensor
    post_cells: torch.Tensor
    post_branches: torch.Tensor
    successor_posts: torch.Tensor
    successor_branches: torch.Tensor
    pieces: int


@dataclass(frozen=True)
class Game:
    """The game DAGs of a batch of obligations, sharing one layered representation."""

    obligations: tuple
    layers: tuple
    transitions: tuple
    losing_version: int

    def to(self, device):
        def move(item):
            return replace(item, **{name: value.to(device) for name, value in vars(item).items()
                                    if torch.is_tensor(value)})
        return replace(self, layers=tuple(map(move, self.layers)), transitions=tuple(map(move, self.transitions)))


@dataclass(frozen=True)
class RepairConstraint:
    """``gamma = (G, i)``: the winning continuation found for one obligation.

    ``game`` holds the DAG (shared with the batch), ``winning`` the winning
    action slots per depth, ``index`` the obligation within the batch.
    Anchored constraints (root at time 0) can never be backtracked.
    """

    game: Game
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


# --------------------------------------------------------------- backends

# A backend W proposes native values per game node: ``propose(boxes, cells,
# depth, policy) -> (N, K, action_dim)``. The values offered on a cell are
# fixed for the whole run, so losing boxes and whole losing cells are certified
# relative to W. ``fixed``: the current value is not offered in addition.

class Alphabet:
    """Exhaustive backend: the same native values at every node, plus the current one."""

    fixed = False

    def __init__(self, values):
        self.values = torch.as_tensor(values)
        if self.values.ndim != 2 or not len(self.values):
            raise ValueError("an alphabet is a nonempty (actions, action_dim) tensor of native values")

    def propose(self, boxes, cells, depth, policy):
        return self.values.to(policy.lower)[None].expand(len(boxes), -1, -1)


class LocalAlphabet:
    """Corrections of the network's own value (Car2D): at every node the
    unrepaired network's native value of the node's cell shifted by ``0, +-d``
    for ``d`` in ``deltas`` along the action dimensions ``dims`` (all if
    ``None``; the others keep the network value), clamped to the repair range,
    nearest first, plus the current value. A delta as large as the range
    reaches its ends from any value.

    The values offered on a cell are fixed for the whole run, so failing all of
    them certifies losing boxes, and the games of different obligations offer
    the same values on a cell.
    """

    fixed = False

    def __init__(self, policy, deltas, dims=None):
        span = policy.max_value - policy.min_value
        deltas = torch.tensor([int(d) for d in deltas])
        axis = torch.unique(torch.cat((-deltas, deltas, deltas.new_zeros(1))).clamp(-span, span))
        dims = range(policy.action_dim) if dims is None else [int(d) for d in dims]
        axes = [axis if d in dims else axis.new_zeros(1) for d in range(policy.action_dim)]
        offsets = torch.cartesian_prod(*axes).reshape(-1, policy.action_dim).float()
        self.offsets = offsets[torch.argsort(offsets.square().sum(1), stable=True)]

    def propose(self, boxes, cells, depth, policy):
        origin = policy.network_values(cells)
        return (origin[:, None] + self.offsets.to(origin)[None]).clamp(policy.min_value, policy.max_value)


class Backup:
    """A fixed backup controller: exactly one native value per node, the
    current controller's value is not offered. Losing is relative to the backup."""

    fixed = True

    def __init__(self, command):
        self.command = command

    def propose(self, boxes, cells, depth, policy):
        return self.command(boxes)[:, None, :]


def constant_backup(policy, endpoint, *, expect=None):
    """The ``"min"`` or ``"max"`` end of a scalar native range; ``expect`` checks its decoded action."""
    if policy.action_dim != 1 or endpoint not in ("min", "max"):
        raise ValueError("a constant backup needs a scalar action and endpoint 'min' or 'max'")
    endpoints = policy.repair_candidates(2)
    decoded = policy.decode(endpoints)[:, 0]
    index = int(decoded.argmax() if endpoint == "max" else decoded.argmin())
    if expect is not None and not torch.isclose(decoded[index], decoded.new_tensor(float(expect)), rtol=0, atol=1e-6):
        raise ValueError(f"the native range cannot express the backup action {expect}")
    value = endpoints[index].clone()
    return Backup(lambda boxes: value.to(boxes).expand(len(boxes), -1))


def cell_centre_backup(policy, command):
    """A state-feedback ``command(states) -> actions`` evaluated at the centre of
    each policy cell (outer cells closed at the grid bounds) and encoded as the
    native value whose decoded action is nearest, per component."""
    axis = torch.arange(policy.min_value, policy.max_value + 1, dtype=torch.float32, device=policy.device)
    decoded = policy.decode(torch.stack([axis] * policy.action_dim, 1))

    def centres(boxes):
        middle = 0.5 * (boxes[:, 0] + boxes[:, 1])
        cells = policy.cell_ids(middle)
        result = middle.clone()
        for d, row in enumerate(policy.thresholds):
            finite = row[:-1].to(middle)
            index = ((cells // int(policy.strides[d])) % int(policy.sizes[d])).clamp(max=len(finite))
            lower = torch.cat((policy.lower[d:d + 1].to(middle), finite))[index]
            upper = torch.cat((finite, policy.upper[d:d + 1].to(middle)))[index]
            result[:, d] = 0.5 * (lower + upper)
        return result

    def native(boxes):
        target = torch.as_tensor(command(centres(boxes.double())), dtype=torch.float64).clamp(-1.0, 1.0)
        table = decoded.to(target)
        return torch.stack([axis.to(target.device)[(table[:, d][None] - target[:, d][:, None]).abs().argmin(1)]
                            for d in range(target.shape[1])], 1).to(boxes.dtype)

    return Backup(native)


def docking_approach_backup(problem, policy, speed=0.15):
    """Docking2D: thrust towards the target velocity ``speed`` pointing at the dock."""
    config = problem.system.config

    def command(states):
        position, velocity = states[:, :2], states[:, 2:]
        norm = position.norm(dim=-1, keepdim=True)
        direction = torch.where(norm > 1e-9, -position / norm.clamp_min(1e-9), torch.zeros_like(position))
        return (float(speed) * direction - velocity) * config.mass / config.dt

    return cell_centre_backup(policy, command)


def load_backend(path, problem, policy):
    """The backend described by a JSON file::

        {"type": "game", "action_samples": 11}
        {"type": "constant-backup", "endpoint": "max", "expect": 1.0}
        {"type": "docking-approach", "speed": 0.15}
        {"type": "local", "deltas": [2, 8, 32, 256], "dims": [1]}
    """
    config = json.loads(Path(path).read_text())
    kind = config["type"]
    if kind == "game":
        return Alphabet(policy.repair_candidates(int(config.get("action_samples", 11))))
    if kind == "constant-backup":
        return constant_backup(policy, config.get("endpoint", "max"), expect=config.get("expect"))
    if kind == "docking-approach":
        return docking_approach_backup(problem, policy, float(config.get("speed", 0.15)))
    if kind == "local":
        return LocalAlphabet(policy, config["deltas"], config.get("dims"))
    raise ValueError(f"unknown backend type {kind!r}")


# --------------------------------------------------------------- FindCont

@dataclass(frozen=True)
class FindContResult:
    """Constraints for the obligations that won, the obligations without a safe
    continuation, the boxes added to ``L``, and timing/size statistics."""

    constraints: tuple
    unresolved: tuple
    losing_delta: torch.Tensor
    statistics: dict


@torch.no_grad()
def find_continuations(obligations, backend, problem, policy, losing_set, *, learn=True):
    """FindCont for a batch of obligations: one game per horizon, shortest first.

    ``losing_set`` grows in place when ``learn``; the games of longer
    horizons already see the growth of shorter ones.

    In Algorithm 2 of the paper, we update L outside the continuation search
    here we do it inside such that later calls can already benefit from expanded
    losing set knowledge. The returned ``losing_delta`` holds the pieces that
    actually enlarged ``L`` in this call; the caller needs it for
    EarliestAffected, not to add it again.
    """
    if any(o.horizon < 1 for o in obligations):
        raise ValueError("every obligation needs a positive horizon")
    pending, constraints, deltas = list(obligations), [], []
    statistics = {"expansion_seconds": 0.0, "solving_seconds": 0.0, "nodes": 0, "pieces": 0}
    for horizon in sorted({o.horizon for o in obligations}):
        group = tuple(o for o in obligations if o.horizon == horizon)
        _synchronize(policy)
        started = perf_counter()
        game = _expand(group, backend, problem, policy, losing_set)
        _synchronize(policy)
        statistics["expansion_seconds"] += perf_counter() - started
        started = perf_counter()
        winning, won, delta = _solve(game, backend, problem, policy, losing_set,
                                     learn=learn)
        _synchronize(policy)
        statistics["solving_seconds"] += perf_counter() - started
        statistics["nodes"] += sum(len(layer.boxes) for layer in game.layers)
        statistics["pieces"] += sum(t.pieces for t in game.transitions)
        game = game.to("cpu")
        winning = tuple(mask.cpu() for mask in winning)
        for index in torch.where(won)[0].tolist():
            constraints.append(RepairConstraint(game, winning, index))
        won_set = {group[i] for i in torch.where(won)[0].tolist()}
        pending = [o for o in pending if o not in won_set]
        deltas.append(delta.cpu())
    delta = torch.cat(deltas) if deltas else torch.empty((0, 2, len(policy.thresholds)), dtype=policy.lower.dtype)
    return FindContResult(tuple(constraints), tuple(pending), delta, statistics)


def _synchronize(policy):
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)


def _expand(obligations, backend, problem, policy, losing_set):
    """Expand the game DAGs of obligations sharing one horizon up to that horizon."""
    spec, device, dtype = problem.spec, policy.device, policy.lower.dtype
    roots = torch.stack([o.box for o in obligations]).to(device=device, dtype=dtype)
    goals = spec.goal_boxes.to(roots)
    active, parents, _, reached = cut_goal(roots, goals, ~spec.intersects_unsafe(roots))
    # A root entirely inside the goal stays as a terminal node.
    completed = reached & (torch.bincount(parents, minlength=len(roots)) == 0)
    active = torch.cat((active, roots[completed]))
    parents = torch.cat((parents, torch.where(completed)[0]))
    recorded = torch.tensor([o.cell for o in obligations], dtype=torch.long, device=device)[parents]
    known = recorded >= 0
    # Keep the recorded cell of an already aligned root: projecting again would
    # also split off its faces into neighbouring cells.
    # shouldnt be necessary
    for d, row in enumerate(policy.thresholds):
        coordinate = (recorded[known] // policy.strides[d]) % policy.sizes[d]
        low = torch.cat((row.new_tensor([-torch.inf]), row))[coordinate]
        if bool(((active[known][:, 0, d] < low) | (active[known][:, 1, d] > row[coordinate])).any()):
            raise ValueError("an obligation's box lies outside its recorded policy cell")
    pieces, piece_cells, piece_parents = cell_pieces(active[~known], policy)
    boxes = torch.cat((active[known], pieces))
    cells = torch.cat((recorded[known], piece_cells))
    owners = torch.cat((parents[known], parents[~known][piece_parents]))
    noninitial = torch.tensor([o.time > 0 for o in obligations], device=device)
    if bool((spec.intersects_unsafe(boxes) & noninitial[owners]).any()):
        raise ValueError("an obligation's box after time 0 is already unsafe")
    layers = [GameLayer(boxes, cells, owners, losing_set.intersects(boxes, cells))]
    horizon = obligations[0].horizon
    transitions = []
    for depth in range(horizon):
        transition, layer = _expand_layer(layers[-1], depth, depth + 1 >= horizon, backend, problem, policy,
                                          losing_set)
        transitions.append(transition)
        if layer is None:
            break
        layers.append(layer)
    return Game(tuple(obligations), tuple(layers), tuple(transitions), losing_set.version)


def _expand_layer(layer, depth, last, backend, problem, policy, losing_set):
    """Apply every proposed action at every node of a layer; returns the
    transition and the next layer (``None`` if no branch continues)."""
    spec = problem.spec
    nodes_count = len(layer.boxes)
    values = backend.propose(layer.boxes, layer.cells, depth, policy)
    if not backend.fixed:
        # also include the current policy cells value
        values = torch.cat((values, policy.cell_values(layer.cells)[:, None, :]), 1)
    slots = values.shape[1]
    shape, device = (nodes_count, slots), values.device
    expanded, terminal, safe = (torch.zeros(shape, dtype=torch.bool, device=device) for _ in range(3))
    goal_reached, terminal_intersects, terminal_contained = (torch.zeros_like(expanded) for _ in range(3))
    kept_boxes, kept_cells, kept_branches, children = [], [], [], []
    kept_count = pieces_count = 0
    flat_values = values.reshape(-1, values.shape[-1])

    # A node inside the goal (and not in U or L) is a terminal win.
    in_goal = inside_any(layer.boxes, spec.goal_boxes.to(layer.boxes))
    in_goal &= ~spec.intersects_unsafe(layer.boxes) & ~layer.losing
    for mask in (expanded, terminal, safe, goal_reached):
        mask[in_goal] = True

    for start in range(0, nodes_count * slots, BRANCH_BATCH):
        branches = torch.arange(start, min(start + BRANCH_BATCH, nodes_count * slots), device=device)
        branches = branches[~layer.losing[branches // slots] & ~in_goal[branches // slots]]
        if not len(branches):
            continue
        nodes = branches // slots
        boxes = layer.boxes[nodes]
        actions = policy.decode(flat_values[branches])
        tube_ok = tube_safe(problem, boxes, actions)
        safe.view(-1)[branches] = tube_ok
        expanded.view(-1)[branches] = True
        # An uncertified tube cannot win, but its post-image is still followed:
        # only its containment in L lets the node be certified losing.
        terminal.view(-1)[branches] = last | ~tube_ok
        remaining = torch.zeros(len(branches), dtype=torch.long, device=device)
        reached, meets, outside = torch.zeros_like(remaining), torch.zeros_like(remaining), torch.zeros_like(remaining)
        for posts, cells, parents, _, reached_parents in successors(problem, policy, boxes, actions, tube_ok,
                                                                    piece_budget=PIECE_BUDGET):
            reached.index_add_(0, reached_parents, torch.ones_like(reached_parents))
            remaining.index_add_(0, parents, torch.ones_like(parents))
            intersects = losing_set.intersects(posts, cells)
            contained = losing_set.contains(posts, cells)
            meets.index_add_(0, parents, intersects.long())
            outside.index_add_(0, parents, (~contained).long())
            pieces_count += len(posts)
            owner = branches[parents]
            keep = ~terminal.view(-1)[owner]
            if bool(keep.any()):
                children.append(torch.where(~intersects[keep])[0] + kept_count)
                kept_boxes.append(posts[keep])
                kept_cells.append(cells[keep])
                kept_branches.append(owner[keep])
                kept_count += int(keep.sum())
        goal_reached.view(-1)[branches] = reached > 0
        # Reaching the goal with nothing left over ends a branch successfully.
        terminal.view(-1)[branches] |= (remaining == 0) & (reached > 0)
        terminal_intersects.view(-1)[branches] = meets > 0
        terminal_contained.view(-1)[branches] = (remaining > 0) & (outside == 0) & (reached == 0)

    empty = torch.empty(0, dtype=torch.long, device=device)
    post_boxes = torch.cat(kept_boxes) if kept_boxes else layer.boxes.new_empty((0, *layer.boxes.shape[1:]))
    post_cells = torch.cat(kept_cells) if kept_cells else empty
    post_branches = torch.cat(kept_branches) if kept_branches else empty
    successor_posts = torch.cat(children) if children else empty
    # Terminal branches keep no pieces; only their summaries matter.
    summarized = terminal
    transition = GameTransition(
        values=values, expanded=expanded, terminal=terminal, safe_tubes=safe, goal_reached=goal_reached,
        terminal_intersects=terminal_intersects & summarized, terminal_contained=terminal_contained & summarized,
        post_boxes=post_boxes, post_cells=post_cells, post_branches=post_branches,
        successor_posts=successor_posts, successor_branches=post_branches[successor_posts], pieces=pieces_count)
    if not len(successor_posts):
        return transition, None
    next_layer = GameLayer(post_boxes[successor_posts], post_cells[successor_posts],
                           layer.owners[post_branches[successor_posts] // slots],
                           torch.zeros(len(successor_posts), dtype=torch.bool, device=device))
    return transition, next_layer


def _solve(game, backend, problem, policy, losing_set, *, learn):
    """Winning slots per depth, which obligations won, and the boxes certified losing.

    Labels are computed against ``L`` as it was during expansion; the boxes
    certified here are added to ``L`` once at the end.
    """
    # this is mainly a leftover from a previous prototype, could be simplified further such that
    # losing set is no longer passed even.
    if game.losing_version != losing_set.version:
        raise ValueError("a game must be solved against the losing set it was expanded with")
    winning, losing_nodes = [None] * len(game.transitions), [None] * len(game.transitions)
    winning_next = losing_next = None
    for depth in range(len(game.transitions) - 1, -1, -1):
        transition, layer = game.transitions[depth], game.layers[depth]
        nodes_count, slots = transition.values.shape[:2]
        branches = nodes_count * slots
        expanded, terminal = transition.expanded.reshape(-1), transition.terminal.reshape(-1)
        post_meets = losing_set.intersects(transition.post_boxes, transition.post_cells)
        # branch meets losing set?
        branch_meets = _any_per(post_meets, transition.post_branches, branches)
        branch_meets |= transition.terminal_intersects.reshape(-1)
        branch_winning = expanded & terminal & ~branch_meets
        continuing = expanded & ~terminal & ~branch_meets
        if bool(continuing.any()):
            children = torch.zeros(branches, dtype=torch.long, device=expanded.device)
            winning_children = torch.zeros_like(children)
            children.index_add_(0, transition.successor_branches, torch.ones_like(transition.successor_branches))
            winning_children.index_add_(0, transition.successor_branches, winning_next.long())
            if bool((continuing & (children == 0)).any()):
                raise ValueError("a continuing branch has no successor nodes")
            # only become winnign if all children are winning (quantifier alternation in the paper)
            branch_winning |= continuing & (winning_children == children)
        slot_winning = branch_winning.reshape(nodes_count, slots) & transition.safe_tubes
        winning[depth] = slot_winning
        winning_next = slot_winning.any(1) & ~losing_set.intersects(layer.boxes, layer.cells)
        if not learn:
            continue
        # A piece is losing if it lies in L or became a losing node of the next layer.
        post_losing = losing_set.contains(transition.post_boxes, transition.post_cells)
        if losing_next is not None:
            post_losing[transition.successor_posts] |= losing_next
        branch_losing = _all_per(post_losing, transition.post_branches, branches)
        branch_losing = torch.where(transition.terminal.reshape(-1), transition.terminal_contained.reshape(-1),
                                    branch_losing)
        branch_losing &= ~transition.goal_reached.reshape(-1)
        node_losing = (expanded & branch_losing).reshape(nodes_count, slots).all(1)
        losing_nodes[depth] = node_losing
        losing_next = losing_set.contains(layer.boxes, layer.cells) | node_losing

    roots = game.layers[0]
    count = torch.bincount(roots.owners, minlength=len(game.obligations))
    winning_count = torch.zeros_like(count).index_add_(0, roots.owners, winning_next.long())
    won = (count > 0) & (count == winning_count)

    certified = []
    if learn:
        by_cell = {}
        for layer, mask in zip(game.layers, losing_nodes):
            for cell in torch.unique(layer.cells[mask]).tolist():
                by_cell.setdefault(cell, []).append(layer.boxes[mask & (layer.cells == cell)])
        for cell, groups in by_cell.items():
            boxes = torch.cat(groups)
            # attempt the whole cell is losing
            if _cell_losing(cell, backend, problem, policy, losing_set):
                certified.append(losing_set.cell_box(cell)[None])
            else:
                known = losing_set.contains(boxes, torch.full((len(boxes),), cell, dtype=torch.long,
                                                               device=boxes.device))
                certified.append(boxes[~known].cpu())
    certified = [boxes for boxes in certified if len(boxes)]
    delta = losing_set.add(torch.cat(certified)) if certified else \
        torch.empty((0, 2, len(policy.thresholds)), dtype=policy.lower.dtype)
    return tuple(winning), won, delta


def _any_per(values, owners, count):
    hits = torch.zeros(count, dtype=torch.long, device=values.device)
    return hits.index_add_(0, owners, values.long()) > 0


def _all_per(values, owners, count):
    total = torch.zeros(count, dtype=torch.long, device=values.device).index_add_(0, owners, torch.ones_like(owners))
    hits = torch.zeros_like(total).index_add_(0, owners, values.long())
    return (total > 0) & (hits == total)


def _cell_losing(cell, backend, problem, policy, losing_set):
    """Whether the whole policy cell is losing: every value the backend offers
    on the cell leads entirely into ``L`` without touching the goal."""
    box = losing_set.cell_box(cell)
    if box is None:
        return False
    box = box.to(policy.lower)
    cells = torch.tensor([cell], device=box.device)
    values = backend.propose(box[None], cells, 0, policy)[0]
    if not backend.fixed:
        values = torch.cat((values, policy.cell_values(cells)))
    for chunk in values.split(BRANCH_BATCH):
        boxes = box[None].expand(len(chunk), -1, -1)
        everywhere = torch.ones(len(chunk), dtype=torch.bool, device=box.device)
        for posts, cells, _, _, reached in successors(problem, policy, boxes, policy.decode(chunk), everywhere,
                                                      piece_budget=PIECE_BUDGET):
            if len(reached) or not bool(losing_set.contains(posts, cells).all()):
                return False
    return True
