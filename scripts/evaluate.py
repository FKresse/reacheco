"""Rollout evaluation of the original and the repaired controllers: ACC (Table 5), Car2D and Docking2D (Table 3).

Run from the repository root::

    python scripts/evaluate.py acc          # results/acc/invariant/p100/seed<s>/evaluation.json
    python scripts/evaluate.py car2d        # results/car2d/wall_box_b_1x1/evaluation.json
    python scripts/evaluate.py docking2d    # results/docking2d/seed<s>/evaluation.json (needs SafeRL)

The original and the repaired controller of a run fly the same starts
(``--episodes``, default 1,000, drawn with ``--seed``) to the end of the episode:

    acc        X0 cut at Pmax 100, 10% at its frontier (``ACC(mode="evaluate")``), the
               L* repairs; 410 steps and ACC's reward; unsafe: a sampled collision
    car2d      uniform in X0 (box B) until the goal line x >= 6, a crash or 100 steps;
               the course reward; unsafe: an obstacle or the world, heading or speed limits
    docking2d  from the 130 m circle at rest (``circle``, the training starts) and from
               X0 (``x0``) in SafeRL's training environment (``_saferl.py``) for up to
               5,000 steps; its reward; unsafe: a step above the verified velocity limit
               (ARCH-COMP's ``0.2 + 2 n ||p||``; the episode goes on). SafeRL's own limit,
               ``0.2 + 2 n max(||p|| - 0.5, 0)``, is up to 0.001 m/s stricter (``saferl_unsafe``)

Rollouts check sampled states only: empirical safety, not a certificate. The
experiment scripts' ``report`` commands show the results.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _runs import report  # noqa: E402

BENCHMARKS = ("acc", "car2d", "docking2d")
ACC_PROBLEM = "problems/acc/boundary160_invariant_100.json"
CIRCLE_RADIUS = 130.0
DOCKING_PROBLEM = "problems/docking2d/box70_120.json"
DOCKING_STEPS = 5000


class Actor:
    """A controller as ``state -> action``, its actions cached per policy cell."""

    def __init__(self, path):
        from reacheco.policy import DWCPolicy
        self.policy, self.actions = DWCPolicy.load(path), {}

    @torch.no_grad()
    def __call__(self, state):
        cell = int(self.policy.cell_ids(torch.as_tensor(state, dtype=torch.float32)[None])[0])
        if cell not in self.actions:
            self.actions[cell] = self.policy.decode(self.policy.cell_values([cell]))[0].numpy()
        return self.actions[cell]


def system(benchmark):
    """The benchmark's gym environment in evaluation mode, and the ``info`` key that flags a step unsafe."""
    from reacheco.environments.acc import ACC
    from reacheco.environments.car2d import Car2D
    from reacheco.spec import load_problem

    if benchmark == "acc":
        problem = load_problem(ROOT / ACC_PROBLEM)
        return ACC(mode="evaluate", initial_boxes=problem.spec.initial_boxes, boundary_probability=0.1), "crash"
    import car2d
    problem = load_problem(ROOT / car2d.PROBLEM)
    config = replace(problem.system.config, max_steps=problem.spec.horizon)
    return Car2D(config, mode="evaluate", initial_boxes=problem.spec.initial_boxes,
                 goal_boxes=problem.spec.goal_boxes), "unsafe"


def gym_episodes(benchmark, path, episodes, seed):
    """ACC or Car2D episodes; the environment draws the starts, from ``seed``."""
    env, unsafe = system(benchmark)
    act, records = Actor(path), []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        record = {"start": state.tolist(), "reward": 0.0, "steps": 0, "unsafe": False}
        terminated = truncated = False
        while not (terminated or truncated):
            state, reward, terminated, truncated, info = env.step(act(state))
            record["reward"] += reward
            record["steps"] += 1
            record["unsafe"] |= bool(info[unsafe])
        records.append(record)
    return records


def docking_starts(kind, episodes, seed):
    """At rest on the 130 m circle (uniform angle), or uniform in X0."""
    rng = np.random.default_rng(seed)
    if kind == "circle":
        angle = rng.uniform(0.0, 2.0 * np.pi, episodes)
        return np.stack((CIRCLE_RADIUS * np.cos(angle), CIRCLE_RADIUS * np.sin(angle),
                         np.zeros(episodes), np.zeros(episodes)), axis=1).astype(np.float32)
    from reacheco.spec import load_problem
    (lower, upper), = load_problem(ROOT / DOCKING_PROBLEM).spec.initial_boxes.numpy()
    return rng.uniform(lower, upper, size=(episodes, 4)).astype(np.float32)


def docking_episodes(kind, path, episodes, seed):
    from _saferl import Docking
    from reacheco.spec import load_problem
    env, act, records = Docking(), Actor(path), []
    spec = load_problem(ROOT / DOCKING_PROBLEM).spec
    for start in docking_starts(kind, episodes, seed):
        state = env.reset(start)
        record = {"start": start.tolist(), "reward": 0.0, "steps": 0, "unsafe": False, "saferl_unsafe": False,
                  "docked": False}
        for _ in range(DOCKING_STEPS):
            state, reward, done, info = env.step(act(state))
            record["reward"] += reward
            record["steps"] += 1
            record["unsafe"] |= bool(spec.inside_unsafe(torch.as_tensor(state, dtype=torch.float64).expand(1, 2, 4)))
            record["saferl_unsafe"] |= bool(info.get("status", {}).get("max_vel_violation", -np.inf) > 0)
            if done:
                record["docked"] = bool(info.get("success"))
                break
        records.append(record)
    return records


def summary(records):
    rewards = np.array([r["reward"] for r in records])
    result = {"episodes": len(records), "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std(ddof=1)),
              "unsafe_percent": 100.0 * float(np.mean([r["unsafe"] for r in records])),
              "steps_mean": float(np.mean([r["steps"] for r in records]))}
    if "docked" in records[0]:
        result["docked_percent"] = 100.0 * float(np.mean([r["docked"] for r in records]))
        result["saferl_unsafe_percent"] = 100.0 * float(np.mean([r["saferl_unsafe"] for r in records]))
    return {**result, "records": records}


def tasks(benchmark, results, seeds):
    """``(output, start set, controller name, controller)`` per rollout set; runs without a certificate are left out."""
    if benchmark == "acc":
        import acc
        runs = [(acc.output(results, "invariant", 100, s), ROOT / f"checkpoints/acc/dwc_b81_seed{s}.pt", None)
                for s in seeds or acc.SEEDS]
    elif benchmark == "car2d":
        import car2d
        runs = [(results / "wall_box_b_1x1", ROOT / car2d.CHECKPOINT, None)]
    else:
        import docking2d
        runs = [(results / f"seed{s}", docking2d.checkpoint(s), ("circle", "x0")) for s in seeds or docking2d.SEEDS]
    found = []
    for output, original, kinds in runs:
        if (report(output) or {}).get("status") != "safe":
            print(f"{output}: not certified, skipped", flush=True)
            continue
        for kind in kinds or (None,):
            found += [(output, kind, "original", original), (output, kind, "repaired", output / "controller.pt")]
    return found


def rollout(benchmark, kind, path, episodes, seed):
    torch.set_num_threads(1)
    if benchmark == "docking2d":
        return summary(docking_episodes(kind, path, episodes, seed))
    return summary(gym_episodes(benchmark, path, episodes, seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--results", type=Path, help="the benchmark's results (default: results/<benchmark>)")
    parser.add_argument("--seeds", nargs="+", type=int, help="controller seeds (default: all)")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="seed of the starts")
    parser.add_argument("--jobs", type=int, default=4, help="rollout sets at a time (one CPU each)")
    arguments = parser.parse_args()
    results = arguments.results or ROOT / "results" / arguments.benchmark

    todo = tasks(arguments.benchmark, results, arguments.seeds)
    with ProcessPoolExecutor(arguments.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(rollout, arguments.benchmark, kind, path, arguments.episodes, arguments.seed)
                   for _, kind, _, path in todo]
        evaluations = {}
        for (output, kind, name, path), future in zip(todo, futures):
            result = {"controller": str(path), **future.result()}
            entry = evaluations.setdefault(output, {"episodes": arguments.episodes, "seed": arguments.seed})
            (entry.setdefault(kind, {}) if kind else entry)[name] = result
            print(f"{output} {kind or ''} {name}: reward {result['reward_mean']:.3f} +- {result['reward_std']:.3f}, "
                  f"unsafe {result['unsafe_percent']:.1f}%", flush=True)
    for output, entry in evaluations.items():
        (output / "evaluation.json").write_text(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
