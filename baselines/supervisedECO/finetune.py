"""BBReach + Supervised (Appendix E): counterexample-guided fine-tuning of an ACC DWC.

Run from the repository root::

    python baselines/supervisedECO/finetune.py --seed 1 --output results/supervised/seed1

1. Collect 100,000 clean transitions of the original controller (start and
   successor outside the braking invariant's unsafe set, tube certified);
   their actions are the preservation targets.
2. Run bounded reachability (``--horizon`` steps, merging as in ReachECO) from
   ``X_0``, removing failing boxes from the frontier. For every failure, the
   failing boxes' parents are replayed without merging; each source box whose
   successor lies entirely in the unsafe set (and which is not unsafe itself)
   adds its cell's upper corner with target ``--target`` (1.05, beyond full
   braking: the regression is unclipped) to the repair data.
3. Train 1000 steps (batch 256, half from each dataset, Adam 3e-4), export the
   hard controller, and repeat from 2 until a clean pass or ``--rounds``.
4. Verify the final controller with ReachECO's forward pass on the full
   problem (horizon 200, fixed point), without repair.

The report's ``seconds`` covers steps 1-4.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from baselines.supervisedECO.model import load  # noqa: E402
from reacheco.boxes import cell_pieces, convert, cut_goal  # noqa: E402
from reacheco.environments.acc import ACC  # noqa: E402
from reacheco.losing import LosingSet  # noqa: E402
from reacheco.reach import reach_step  # noqa: E402
from reacheco.spec import load_problem  # noqa: E402
from reacheco.tubes import tube_safe  # noqa: E402
from reacheco.verify import verify  # noqa: E402
from acc import action_change  # noqa: E402


@torch.no_grad()
def clean_transitions(policy, problem, count, seed, envs=32):
    """``count`` clean (state, action) pairs of the controller.

    Starts ``p ~ U(0.001, 100)`` and ``v ~ U(-sqrt(2 B p) + 1e-3, min(0, sqrt(2 A (200 - p)) - 1e-3))``
    (recoverable and closing in); an episode restarts at its first unclean transition.
    """
    spec, rng = problem.spec, np.random.default_rng(seed)
    systems = [ACC(problem.system.config) for _ in range(envs)]
    config = problem.system.config

    def start():
        position = rng.uniform(0.001, 100.0)
        low = -np.sqrt(2.0 * config.max_braking_acceleration * position) + 1e-3
        high = min(np.sqrt(2.0 * config.max_acceleration * (config.max_distance - position)) - 1e-3, 0.0)
        return np.array([position, rng.uniform(low, max(low, high))], dtype=np.float32)

    for system in systems:
        system.state, system.steps = start(), 0
    points, actions, attempted = [], [], 0
    while sum(map(len, points)) < count:
        states = torch.from_numpy(np.stack([system.state for system in systems])).to(policy.device)
        applied = policy.decode(policy.cell_values(policy.cell_ids(states)))
        results = [system.step(action.cpu().numpy()) for system, action in zip(systems, applied)]
        successors = torch.from_numpy(np.stack([result[0] for result in results])).to(policy.device)
        boxes, next_boxes = states[:, None].expand(-1, 2, -1), successors[:, None].expand(-1, 2, -1)
        clean = ~spec.intersects_unsafe(boxes) & ~spec.intersects_unsafe(next_boxes)
        clean &= tube_safe(problem, boxes.contiguous(), applied)
        points.append(states[clean].cpu())
        actions.append(applied[clean].cpu())
        attempted += envs
        for system, ok, (_, _, terminated, truncated, _) in zip(systems, clean.tolist(), results):
            if not ok or terminated or truncated:
                system.state, system.steps = start(), 0
        if attempted >= max(10_000, 100 * count):
            raise RuntimeError(f"only {sum(map(len, points))} of {count} clean transitions")
    return {"points": torch.cat(points)[:count], "targets": torch.cat(actions)[:count], "attempted": attempted}


@torch.no_grad()
def explore(policy, problem, horizon, target):
    """Bounded reachability with failing boxes removed; returns the repair data and a report."""
    spec = problem.spec
    current = convert(spec.initial_boxes, torch.float32, outward=True).to(policy.device)
    unsafe = spec.intersects_unsafe(current)
    failures, current = int(unsafe.sum()), current[~unsafe]
    goals = convert(spec.goal_boxes, torch.float32, outward=False).to(policy.device)
    meets = LosingSet(problem, policy).intersects       # L = U
    points, steps = [], 0
    for step in range(1, horizon + 1):
        if not len(current):
            break
        successors, record = reach_step(current, policy, problem, meets)
        failing = record.failing
        if bool(failing.any()):
            failures += int(failing.sum())
            rows = torch.where(failing)[0]
            parents = torch.unique(record.source_parents[record.piece_sources[torch.isin(record.piece_rows, rows)]])
            boxes = current[parents.to(current.device)]
            # Replay without merging: every row is one piece with a unique source box.
            posts, local = reach_step(boxes, policy, problem, meets, merge=False, lookahead=False)
            active, _, _, _ = cut_goal(boxes, goals, ~spec.intersects_unsafe(boxes))
            sources = cell_pieces(active, policy)[0][local.piece_sources.to(boxes.device)]
            selected = spec.inside_unsafe(posts) & ~spec.intersects_unsafe(sources)
            points.append(sources[selected][:, 1].cpu())
        current = successors[~failing.to(successors.device)]
        steps = step
    points = torch.cat(points) if points else torch.zeros((0, 2))
    data = {"points": points, "targets": torch.full((len(points), 1), float(target))}
    return data, {"status": "safe" if failures == 0 else "unknown", "explored_steps": steps,
                  "failing_boxes": failures, "repair_examples": len(points), "surviving_boxes": len(current)}


def train(model, optimizer, clean, failures, iterations, batch_size):
    """``iterations`` steps of MSE on batches drawn half from each dataset (with replacement)."""
    device = model.thresholds.device
    datasets = [{key: data[key].to(device) for key in ("points", "targets")} for data in (clean, failures)]
    half, losses = batch_size // 2, []
    model.train()
    for _ in range(iterations):
        picks = [torch.randint(len(data["points"]), (half,), device=device) for data in datasets]
        points = torch.cat([data["points"][pick] for data, pick in zip(datasets, picks)])
        targets = torch.cat([data["targets"][pick] for data, pick in zip(datasets, picks)])
        loss = F.mse_loss(model(points), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {"mean_loss": sum(losses) / len(losses), "last_loss": losses[-1]}


def finetune(arguments):
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    problem = load_problem(arguments.problem)
    model = load(ROOT / f"checkpoints/acc/dwc_b81_seed{arguments.seed}.pt",
                 ROOT / f"baselines/supervisedECO/checkpoints/dwc_b81_seed{arguments.seed}.pt")
    model.eval()
    original = policy = model.to_policy()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=arguments.learning_rate)
    # collecting cleaned dataset of transitions
    clean = clean_transitions(policy, problem, arguments.clean_transitions, arguments.seed)
    failures, rounds = None, []
    for index in range(arguments.rounds + 1):
        if index:
            policy = model.to_policy()
        data, report = explore(policy, problem, arguments.horizon, arguments.target)
        rounds.append({"round": index, **report})
        print(f"round {index}: {report}", flush=True)
        if report["status"] == "safe" or index == arguments.rounds or not (len(data["points"]) or failures):
            break
        if len(data["points"]):
            failures = data if failures is None else {k: torch.cat((failures[k], data[k])) for k in data}
        rounds[-1]["training"] = train(model, optimizer, clean, failures, arguments.iterations, arguments.batch_size)
    tuned = perf_counter() - started
    result = verify(problem, policy)
    seconds = perf_counter() - started
    change, delta = action_change(original, policy, 100)
    policy.save(output / "controller.pt")
    report = {"status": "safe" if result.certified else "unknown", "seconds": seconds,
              "finetuning_seconds": tuned, "verification_seconds": seconds - tuned,
              "failure_step": result.failure_step, "rounds": rounds,
              "change_percent": change, "mean_abs_delta": delta, "clean_attempted": clean["attempted"],
              "arguments": {k: str(v) for k, v in vars(arguments).items()}}
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['status']} after {len(rounds)} rounds, {seconds:.0f} s; change {change:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True, help="ACC controller seed (1-5)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem", type=Path, default=ROOT / "problems/acc/boundary160_invariant_100.json")
    parser.add_argument("--horizon", type=int, default=50, help="steps of the bounded exploration")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clean-transitions", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target", type=float, default=1.05)
    arguments = parser.parse_args()
    if arguments.batch_size % 2:
        parser.error("the batch size must be even (half from each dataset)")
    finetune(arguments)


if __name__ == "__main__":
    main()
