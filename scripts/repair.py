"""Repair one controller with ReachECO and save the result.

Run from the repository root::

    python scripts/repair.py --problem problems/acc/boundary160_collision_30.json \\
        --checkpoint checkpoints/acc/dwc_b81_seed1.pt --backend backends/acc_game_11.json \\
        --output results/example

The output directory receives ``controller.pt`` (pruned), ``merge_mask.json``,
``losing_set.json``, ``report.json`` (outcome and one entry per repair) and
``run.json`` (the command line, the ReachECO commit and the GPU).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reacheco.continuation import load_backend  # noqa: E402
from reacheco.losing import LosingSet  # noqa: E402
from reacheco.policy import DWCPolicy  # noqa: E402
from reacheco.repair import repair  # noqa: E402
from reacheco.spec import load_problem  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-losing-set", type=Path, help="a box file {\"boxes\": ...} to start L from")
    parser.add_argument("--no-learn", action="store_true", help="never grow the losing set")
    parser.add_argument("--no-merge", action="store_true", help="keep every successor piece as its own box")
    parser.add_argument("--no-prune", action="store_true", help="keep repairs the certifying pass never read")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args(argv)

    problem = load_problem(arguments.problem)
    policy = DWCPolicy.load(arguments.checkpoint, device=arguments.device)
    backend = load_backend(arguments.backend, problem, policy)
    losing = (LosingSet.load(problem, policy, arguments.initial_losing_set) if arguments.initial_losing_set
              else None)
    arguments.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "reacheco"], capture_output=True,
                           text=True).stdout
    run = {"command": [sys.executable, *sys.argv], "commit": commit.strip(), "uncommitted_changes": bool(dirty),
           "device": torch.cuda.get_device_name(policy.device) if policy.device.type == "cuda" else "cpu"}
    (arguments.output / "run.json").write_text(json.dumps(run, indent=1) + "\n")

    def progress(entry):
        print(f"repair {len(entries)}: failure at step {entry['failure_step']} ({entry['failing']} boxes), "
              f"{entry['status']}, {entry['changed_cells']} cells changed, L +{entry['losing_added']}, "
              f"{entry['seconds']:.1f} s", flush=True)
        entries.append(entry)

    entries = []
    result = repair(problem, policy, backend, losing_set=losing, learn=not arguments.no_learn,
                    merge=not arguments.no_merge, prune=not arguments.no_prune, on_repair=progress)
    result.save(arguments.output)
    print(f"{result.status} (horizon {result.horizon}, reason {result.reason}) after {len(result.repairs)} repairs, "
          f"{result.seconds:.1f} s; {len(policy.changed_cells(result.policy))} changed cells "
          f"({len(policy.changed_cells(result.repaired))} before pruning)", flush=True)


if __name__ == "__main__":
    main()
