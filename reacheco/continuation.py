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
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from reacheco.boxes import cell_pieces, convert, cut_goal, inside_any
from reacheco.constraints import RepairConstraint
from reacheco.reach import successors
from reacheco.tubes import tube_safe

BRANCH_BATCH = 4096
PIECE_BUDGET = 16384


# ------------------------------------------------------------- game DAGs

@dataclass(frozen=True)
class GameLayer:
    """Policy-aligned nodes at one depth: ``boxes``, ``cells``, the obligation
    each node belongs to (``owners``), and whether it meets ``L`` (only roots
    can: deeper nodes are the pieces that do not)."""

    boxes: torch.Tensor
    cells: torch.Tensor
    owners: torch.Tensor
    losing: torch.Tensor


@dataclass(frozen=True)
class GameTransition:
    """The actions tried at the nodes of one layer and where they lead, checked against ``L`` when expanded.

    ``values`` ``(N, K, A)`` are the native values of the ``K`` action slots
    per node. Per branch (node, slot), flattened as ``node * K + slot``:
    ``expanded``, ``terminal`` (horizon reached, tube uncertified, or all
    pieces in the goal), ``safe_tubes``, ``goal_reached``, ``meets`` (a piece
    of its post-image meets ``L``) and ``inside`` (it has pieces, none reached
    the goal, and every piece that ends here lies inside ``L``). A piece ends
    its branch if the branch is terminal or the piece meets ``L``; the others
    become the nodes of the next layer, ``successor_branches`` their branch.
    """

    values: torch.Tensor
    expanded: torch.Tensor
    terminal: torch.Tensor
    safe_tubes: torch.Tensor
    goal_reached: torch.Tensor
    meets: torch.Tensor
    inside: torch.Tensor
    successor_branches: torch.Tensor
    pieces: int


@dataclass(frozen=True)
class Game:
    """The game DAGs of a batch of obligations, sharing one layered representation."""

    obligations: tuple
    layers: tuple
    transitions: tuple


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
        # importantly it needs to be over the cell not over the obligations
        # box, otherwise we get (potentially) unresolvable conflicts.
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
        constraints.extend(_constraints(game, winning, torch.where(won)[0].tolist()))
        won_set = {group[i] for i in torch.where(won)[0].tolist()}
        pending = [o for o in pending if o not in won_set]
        deltas.append(delta.cpu())
    delta = torch.cat(deltas) if deltas else torch.empty((0, 2, len(policy.thresholds)), dtype=policy.lower.dtype)
    return FindContResult(tuple(constraints), tuple(pending), delta, statistics)


def _constraints(game, winning, won):
    """A repair constraint for each won obligation: its nodes of the batch's game DAG, renumbered."""
    parts = {index: ([], [], [], []) for index in won}
    previous = None
    for depth, (layer, transition) in enumerate(zip(game.layers, game.transitions)):
        owners = layer.owners.cpu()
        order = torch.argsort(owners, stable=True)
        counts = torch.bincount(owners, minlength=len(game.obligations))
        position = torch.empty_like(order)
        position[order] = torch.arange(len(order)) - (counts.cumsum(0) - counts)[owners[order]]
        if depth:
            slots = game.transitions[depth - 1].values.shape[1]
            branches = game.transitions[depth - 1].successor_branches.cpu()
            branches = previous[branches // slots] * slots + branches % slots
        cells, values, masks = layer.cells.cpu(), transition.values.cpu(), winning[depth].cpu()
        for index, rows in zip(range(len(counts)), torch.split(order, counts.tolist())):
            if index in parts:
                for part, tensor in zip(parts[index], (cells, values, masks)):
                    part.append(tensor[rows])
                parts[index][3].append(branches[rows] if depth else rows.new_empty(0))
        previous = position
    return [RepairConstraint(game.obligations[index], *map(tuple, parts[index])) for index in won]


def _synchronize(policy):
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)


def _expand(obligations, backend, problem, policy, losing_set):
    """Expand the game DAGs of obligations sharing one horizon up to that horizon."""
    spec, device, dtype = problem.spec, policy.device, policy.lower.dtype
    roots = torch.stack([o.box for o in obligations]).to(device=device, dtype=dtype)
    goals = convert(spec.goal_boxes, roots.dtype, outward=False).to(roots.device)
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
    return Game(tuple(obligations), tuple(layers), tuple(transitions))


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
    expanded, terminal, safe, goal_reached, meets, inside = (torch.zeros(shape, dtype=torch.bool, device=device)
                                                             for _ in range(6))
    child_branches, child_boxes, child_cells = [], [], []
    pieces_count = 0
    flat_values = values.reshape(-1, values.shape[-1])

    # A node inside the goal (and not in U or L) is a terminal win.
    in_goal = inside_any(layer.boxes, convert(spec.goal_boxes, layer.boxes.dtype, outward=False).to(device))
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
        reached, hits, outside = torch.zeros_like(remaining), torch.zeros_like(remaining), torch.zeros_like(remaining)
        for posts, cells, parents, _, reached_parents in successors(problem, policy, boxes, actions, tube_ok,
                                                                    piece_budget=PIECE_BUDGET):
            reached.index_add_(0, reached_parents, torch.ones_like(reached_parents))
            remaining.index_add_(0, parents, torch.ones_like(parents))
            intersects, contained = losing_set.classify(posts, cells)
            pieces_count += len(posts)
            owner = branches[parents]
            # Pieces of terminal branches and pieces meeting L end here; the others become nodes of the next layer.
            ends = terminal.view(-1)[owner] | intersects
            hits.index_add_(0, parents, intersects.long())
            outside.index_add_(0, parents, (ends & ~contained).long())
            if not bool(ends.all()):
                child_branches.append(owner[~ends])
                child_boxes.append(posts[~ends])
                child_cells.append(cells[~ends])
        goal_reached.view(-1)[branches] = reached > 0
        # Reaching the goal with nothing left over ends a branch successfully.
        terminal.view(-1)[branches] |= (remaining == 0) & (reached > 0)
        meets.view(-1)[branches] = hits > 0
        inside.view(-1)[branches] = (remaining > 0) & (outside == 0) & (reached == 0)

    successor_branches = torch.cat(child_branches) if child_branches else torch.empty(0, dtype=torch.long,
                                                                                         device=device)
    transition = GameTransition(
        values=values, expanded=expanded, terminal=terminal, safe_tubes=safe, goal_reached=goal_reached,
        meets=meets, inside=inside, successor_branches=successor_branches, pieces=pieces_count)
    if not len(successor_branches):
        return transition, None
    next_layer = GameLayer(torch.cat(child_boxes), torch.cat(child_cells), layer.owners[successor_branches // slots],
                           torch.zeros(len(successor_branches), dtype=torch.bool, device=device))
    return transition, next_layer


def _solve(game, backend, problem, policy, losing_set, *, learn):
    """Winning slots per depth, which obligations won, and the boxes certified losing.

    Labels use ``L`` as it was during expansion (recorded in the game); the
    boxes certified here are added to ``L`` once at the end.
    """
    winning, losing_nodes = [None] * len(game.transitions), [None] * len(game.transitions)
    winning_next = losing_next = None
    for depth in range(len(game.transitions) - 1, -1, -1):
        transition, layer = game.transitions[depth], game.layers[depth]
        nodes_count, slots = transition.values.shape[:2]
        branches = nodes_count * slots
        expanded, terminal = transition.expanded.reshape(-1), transition.terminal.reshape(-1)
        successor_branches = transition.successor_branches
        children = torch.zeros(branches, dtype=torch.long, device=expanded.device)
        children.index_add_(0, successor_branches, torch.ones_like(successor_branches))
        branch_meets = transition.meets.reshape(-1)
        branch_winning = expanded & terminal & ~branch_meets
        continuing = expanded & ~terminal & ~branch_meets
        if bool(continuing.any()):
            winning_children = torch.zeros_like(children).index_add_(0, successor_branches, winning_next.long())
            if bool((continuing & (children == 0)).any()):
                raise ValueError("a continuing branch has no successor nodes")
            # only become winnign if all children are winning (quantifier alternation in the paper)
            branch_winning |= continuing & (winning_children == children)
        slot_winning = branch_winning.reshape(nodes_count, slots) & transition.safe_tubes
        winning[depth] = slot_winning
        winning_next = slot_winning.any(1) & ~layer.losing
        if not learn:
            continue
        # A branch is losing if the pieces ending there lie in L and the others became losing nodes.
        branch_losing = transition.inside.reshape(-1)
        if losing_next is not None:
            losing_children = torch.zeros_like(children).index_add_(0, successor_branches, losing_next.long())
            branch_losing = branch_losing & (losing_children == children)
        node_losing = (expanded & branch_losing).reshape(nodes_count, slots).all(1)
        losing_nodes[depth] = node_losing
        # Deeper nodes never meet L, so a node is losing exactly when certified here.
        losing_next = node_losing

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
