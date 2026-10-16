"""Docking2D experiment (Table 3, Figure 3): repair the far130 LUT6 controllers from the 130 m start box.

Run from the repository root::

    python scripts/docking2d.py run --seeds 1 2 3 --gpus 1 2 3
    python scripts/docking2d.py report          # results/docking2d/report.md

The start box (x, y in [70, 120], vx, vy in [-0.28, 0.28]) is split into its
four velocity quadrants (``problems/docking2d/box70_120_vx*_vy*.json``). The
quadrants are repaired one after the other, each run continuing from the
controller and the committed constraints of the previous one, with the
greedy restart, the docking-approach backup and without learning L. Passes
over the four quadrants repeat until one pass changes neither the controller
nor the committed constraints. That clean pass verified every quadrant from ``X_0`` with
the final controller, which makes the result a certificate even though the
greedy restart alone is not one. Repairs are not pruned.

Per seed, ``results/docking2d/seed<s>/`` holds one directory per quadrant run
(``pass_<p>/<quadrant>/``), ``progress.json`` (an interrupted run resumes) and,
after the clean pass, the final ``controller.pt``, the merge mask of each
quadrant (``merge_mask_<quadrant>.json``) and ``report.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from _runs import evaluation, markdown, report, rollouts, run_jobs, synthesized_patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QUADRANTS = ("vx0_vy0", "vx0_vy1", "vx1_vy0", "vx1_vy1")
SEEDS = (1, 2, 3)


def checkpoint(seed):
    return ROOT / f"checkpoints/docking2d/far130_lut6_seed{seed}.pt"


def passes(seed, output, max_passes):
    """Repair the quadrants in passes until a clean pass; returns the exit code."""
    from reacheco.continuation import load_backend
    from reacheco.policy import DWCPolicy
    from reacheco.repair import repair
    from reacheco.spec import load_problem

    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    state = (json.loads(progress.read_text()) if progress.exists()
             else {"pass": 0, "quadrant": 0, "clean": True, "policy": None, "committed": None, "runs": []})
    original = DWCPolicy.load(checkpoint(seed), device="cuda")
    policy = DWCPolicy.load(state["policy"], device="cuda") if state["policy"] else original
    committed = torch.load(state["committed"], weights_only=False) if state["committed"] else ()

    def save_state():
        progress.write_text(json.dumps(state, indent=1) + "\n")

    def finish(status, reason=None):
        runs = state["runs"]
        result = {"status": status, "reason": reason, "passes": state["pass"], "seconds": sum(r["seconds"] for r in runs),
                  "repairs": sum(r["repairs"] for r in runs), "overrides": len(policy.overrides),
                  "changed_cells": len(original.changed_cells(policy)), "committed": len(committed), "runs": runs}
        (output / "report.json").write_text(json.dumps(result, indent=1) + "\n")
        return 0 if status == "safe" else 3

    while state["pass"] < max_passes:
        while state["quadrant"] < len(QUADRANTS):
            name = QUADRANTS[state["quadrant"]]
            directory = output / f"pass_{state['pass']:02d}" / name
            problem = load_problem(ROOT / f"problems/docking2d/box70_120_{name}.json")
            backend = load_backend(ROOT / "backends/docking2d_approach.json", problem, policy)
            print(f"pass {state['pass']} {name}: {len(committed)} committed constraints", flush=True)
            # for docking2d we disable learning as it never learns any unsafe set, improving runtime slightly by deactivating it.
            result = repair(problem, policy, backend, learn=False, prune=False, greedy=True, committed=committed,
                            on_repair=lambda entry: print(
                                f"  repair at step {entry['failure_step']}: {entry['status']}, "
                                f"{entry['changed_cells']} cells, {entry['seconds']:.0f} s", flush=True))
            result.save(directory)
            torch.save(result.committed, directory / "committed.pt")
            changed = len(policy.changed_cells(result.repaired))
            added = len(result.committed) - len(committed)
            run = {"pass": state["pass"], "quadrant": name, "status": result.status, "reason": result.reason,
                   "repairs": len(result.repairs), "changed_cells": changed, "committed": len(result.committed),
                   "new_committed": added, "seconds": round(result.seconds, 1)}
            state["runs"].append(run)
            print(f"  {result.status}: {run['repairs']} repairs, {changed} cells, {len(result.committed)} "
                  f"committed constraints (+{added}), {result.seconds:.0f} s", flush=True)
            policy, committed = result.repaired, result.committed
            if result.status != "safe":
                save_state()
                return finish("unknown", f"{name} in pass {state['pass']}: {result.reason}")
            if changed or added:
                state["clean"] = False
            state.update(policy=str(directory / "controller.pt"), committed=str(directory / "committed.pt"),
                         quadrant=state["quadrant"] + 1)
            save_state()
        clean, finished = state["clean"], state["pass"]
        state.update(clean=True, quadrant=0, **{"pass": finished + 1})
        save_state()
        if clean:
            policy.save(output / "controller.pt")
            for name in QUADRANTS:
                shutil.copy(output / f"pass_{finished:02d}" / name / "merge_mask.json", output / f"merge_mask_{name}.json")
            return finish("safe")
    return finish("unknown", f"no clean pass in {max_passes} passes")


def report_tables(results, seeds):
    rows = []
    for seed in seeds:
        r = report(results / f"seed{seed}")
        if r is None:
            rows.append((seed, "–", "–", "–", "–", "–", "–", "–", "–"))
        elif r["status"] == "timeout":
            rows.append((seed, "timeout", "–", "–", "–", "–", "–", "–", "–"))
        else:
            synthesized = synthesized_patch(results / f"seed{seed}") or ("–", "–", "–")
            rows.append((seed, r["status"], r["passes"], r["repairs"], r["changed_cells"], f"{r['seconds']:.0f}",
                         *synthesized))
    text = ("# Docking2D (box 70..120, four velocity quadrants)\n\n"
            + markdown(("seed", "status", "passes", "repairs", "changed cells", "t [s]",
                        "patch logic [LUT]", "patch memory [LUT]", "patch [LUT]"), rows) + "\n")
    rows = []
    for seed in seeds:
        result = evaluation(results / f"seed{seed}") or {}
        circle, x0 = result.get("circle", {}), result.get("x0", {})
        safe = [f"{100 - s['unsafe_percent']:.1f}" if s else "–"
                for s in (circle.get("original"), x0.get("original"), circle.get("repaired"))]
        rows.append((seed, *safe, rollouts(x0.get("original"), 3)[0], rollouts(x0.get("repaired"), 3)[0]))
    text += ("\n## Table 3: rollouts in SafeRL (scripts/evaluate.py docking2d)\n\n"
             + markdown(("seed", "safe [%] X", "safe [%] X0", "safe [%] X after", "reward X0", "reward X0 after"), rows)
             + "\n")
    (results / "report.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "report", "passes"))
    parser.add_argument("--results", type=Path, default=ROOT / "results/docking2d")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--timeout", type=float, default=None, help="seconds per seed (default: none)")
    parser.add_argument("--max-passes", type=int, default=10)
    parser.add_argument("--seed", type=int, help="passes: the controller seed")
    parser.add_argument("--output", type=Path, help="passes: the seed's output directory")
    arguments = parser.parse_args()
    if arguments.command == "run":
        jobs = [(arguments.results / f"seed{seed}", Path(__file__),
                 ["passes", "--seed", seed, "--max-passes", arguments.max_passes]) for seed in arguments.seeds]
        run_jobs(jobs, arguments.gpus, timeout=arguments.timeout)
    elif arguments.command == "report":
        report_tables(arguments.results, arguments.seeds)
    else:
        sys.exit(passes(arguments.seed, arguments.output, arguments.max_passes))


if __name__ == "__main__":
    main()
