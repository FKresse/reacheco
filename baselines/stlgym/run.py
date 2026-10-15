"""STLGym on ACC at Pmax 100 (Table 2, row STLGym; Figure 5): all seeds, then the row and the figure.

Run from the repository root::

    python baselines/stlgym/run.py run --jobs 5
    python baselines/stlgym/run.py report     # results/stlgym/report.md, figure5.{png,json}

Each seed runs ``finetune.py`` (CPU only) into ``results/stlgym/seed<s>/``;
runs resume like the other experiment scripts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _runs import markdown, mean_std, report, run_jobs  # noqa: E402

FINETUNE = Path(__file__).resolve().parent / "finetune.py"
SEEDS = (0, 1, 2, 3, 4)


def figure(results, reports):
    """Figure 5: mean ACC reward and empirical safety on ``X_0`` over training, one colour per seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {seed: [{"step": c["step"], "reward": c["x0"]["reward_mean"], "safe_percent": c["x0"]["safe_percent"]}
                   for c in r["checks"]] for seed, r in reports.items()}
    (results / "figure5.json").write_text(json.dumps(data, indent=1) + "\n")
    figure, reward = plt.subplots(figsize=(6.3, 3.2))
    safety = reward.twinx()
    for index, (seed, points) in enumerate(data.items()):
        colour = f"C{index}"
        steps = [p["step"] for p in points]
        reward.plot(steps, [p["reward"] for p in points], "s-", color=colour, label=f"seed {seed}")
        safety.plot(steps, [p["safe_percent"] for p in points], "o--", color=colour)
    reward.set_xlabel("Training timestep")
    reward.set_ylabel("Mean ACC reward (squares)")
    safety.set_ylabel("Empirically safe [%] (circles)")
    reward.legend(frameon=False, fontsize="small")
    figure.tight_layout()
    figure.savefig(results / "figure5.png", dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--results", type=Path, default=ROOT / "results/stlgym")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--jobs", type=int, default=5, help="seeds at a time (one CPU thread each)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    arguments = parser.parse_args()
    if arguments.command == "run":
        run_jobs([(arguments.results / f"seed{seed}", FINETUNE, ["--seed", seed]) for seed in arguments.seeds],
                 [""], per_gpu=arguments.jobs, timeout=arguments.timeout)
    reports = {seed: report(arguments.results / f"seed{seed}") for seed in arguments.seeds}
    done = {seed: r for seed, r in reports.items() if r and r.get("status") == "trained"}
    rows = [(seed, r["status"] if r else "missing", f"{r['seconds']:.0f}" if r in done.values() else "–",
             f"{r['checks'][-1]['invariant']['unsafe']}/{r['checks'][-1]['invariant']['episodes']}"
             if r in done.values() else "–",
             f"{r['checks'][-1]['x0']['safe_percent']:.1f}" if r in done.values() else "–",
             f"{r['change_percent']:.1f}" if r in done.values() else "–") for seed, r in reports.items()]
    values = list(done.values())
    text = ("## STLGym, ACC Pmax 100 (U only; not verified)\n\n"
            + markdown(("Seed", "Status", "t [s]", "invariant unsafe (final)", "X0 safe [%] (final)", "Change [%]"),
                       rows)
            + f"\n\nTrained {len(values)}/{len(reports)}; t {mean_std([r['seconds'] for r in values])}; "
              f"change {mean_std([r['change_percent'] for r in values], 1)} %; "
              f"mean |Δ| {mean_std([r['mean_abs_delta'] for r in values], 3)}\n")
    arguments.results.mkdir(parents=True, exist_ok=True)
    (arguments.results / "report.md").write_text(text)
    if done:
        figure(arguments.results, done)
    print(text)


if __name__ == "__main__":
    main()
