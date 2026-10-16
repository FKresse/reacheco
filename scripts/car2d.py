"""Car2D experiment (Appendix A.2): repair the wall controller from start box B.

Run from the repository root::

    python scripts/car2d.py run --gpus 0
    python scripts/car2d.py report          # results/car2d/report.md

The run repairs ``checkpoints/car2d/wall_dwc20x20x10x10_seed0.pt`` on
``problems/car2d/wall_box_b_1x1.json`` with the steering corrections of
``backends/car2d_local.json`` (L learned) and writes ``results/car2d/wall_box_b_1x1/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _runs import REPAIR, evaluation, markdown, report, rollouts, run_jobs, synthesized_patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = "checkpoints/car2d/wall_dwc20x20x10x10_seed0.pt"
PROBLEM = "problems/car2d/wall_box_b_1x1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--results", type=Path, default=ROOT / "results/car2d")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds")
    arguments = parser.parse_args()
    output = arguments.results / "wall_box_b_1x1"
    if arguments.command == "run":
        run_jobs([(output, REPAIR, ["--problem", ROOT / PROBLEM, "--checkpoint", ROOT / CHECKPOINT,
                                    "--backend", ROOT / "backends/car2d_local.json"])],
                 arguments.gpus, timeout=arguments.timeout)
        return
    result = report(output)
    if result is None:
        sys.exit(f"no report in {output}; run first")
    if result["status"] == "timeout":
        rows = [("timeout", "–", "–", "–", "–", "–", "–", "–", "–")]
    else:
        depths = [int(key[8:]) for entry in result["repairs"] for key in entry if key.startswith("horizon_")]
        rows = [(result["status"], f"{result['seconds']:.0f}", len(result["repairs"]),
                 result["overrides"] + result["pruned_cells"], result["overrides"], max(depths, default=0),
                 *(synthesized_patch(output) or ("–", "–", "–")))]
    text = ("# Car2D (box B)\n\n"
            + markdown(("status", "t [s]", "repairs", "changed cells", "after pruning", "deepest game",
                        "patch logic [LUT]", "patch memory [LUT]", "patch [LUT]"), rows) + "\n")
    rollout = evaluation(output) or {}
    text += ("\n## Rollouts from X0 (scripts/evaluate.py car2d)\n\n"
             + markdown(("controller", "reward", "unsafe [%]"),
                        [(name, *rollouts(rollout.get(name))) for name in ("original", "repaired")]) + "\n")
    (arguments.results / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
