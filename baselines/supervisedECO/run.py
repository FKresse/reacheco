"""BBReach + Supervised on ACC at Pmax 100 with L* (Table 2): all seeds, then the table row.

Run from the repository root::

    python baselines/supervisedECO/run.py run --gpus 1 2 3 4 5
    python baselines/supervisedECO/run.py report     # results/supervised/report.md

Each seed runs ``finetune.py`` into ``results/supervised/seed<s>/``; runs are
limited to one hour and resume like the other experiment scripts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _runs import markdown, mean_std, report, run_jobs  # noqa: E402

FINETUNE = Path(__file__).resolve().parent / "finetune.py"
SEEDS = (1, 2, 3, 4, 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--results", type=Path, default=ROOT / "results/supervised")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    arguments = parser.parse_args()
    if arguments.command == "run":
        run_jobs([(arguments.results / f"seed{seed}", FINETUNE, ["--seed", seed]) for seed in arguments.seeds],
                 arguments.gpus, timeout=arguments.timeout)
    reports = [report(arguments.results / f"seed{seed}") for seed in arguments.seeds]
    certified = [r for r in reports if r and r["status"] == "safe"]
    rows = [(seed, r["status"] if r else "missing", f"{r['seconds']:.0f}" if r and "seconds" in r else "–",
             len(r["rounds"]) if r and "rounds" in r else "–",
             f"{r['change_percent']:.1f}" if r and "change_percent" in r else "–")
            for seed, r in zip(arguments.seeds, reports)]
    text = ("## BBReach + Supervised, ACC Pmax 100, L*\n\n"
            + markdown(("Seed", "Status", "t [s]", "Rounds", "Change [%]"), rows)
            + f"\n\nCertified {len(certified)}/{len(reports)}; t {mean_std([r['seconds'] for r in certified])}; "
              f"change {mean_std([r['change_percent'] for r in certified], 1)} %; "
              f"mean |Δ| {mean_std([r['mean_abs_delta'] for r in certified], 3)}\n")
    (arguments.results / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
