"""Vivado synthesis of the repaired controllers behind the reported patch sizes.

Run from the repository root (needs Vivado: ``VIVADO_SETTINGS`` or ``--vivado-settings``)::
    export VIVADO_SETTINGS=/tools/Xilinx/Vivado/2022.2/settings64.sh
    python scripts/synthesize.py                      # acc, car2d and docking2d
    python scripts/synthesize.py acc docking2d --jobs 8

For every certified run whose patch the reports list, the controller is
implemented with and without its patch (``circuit.synthesize``): ACC Table 2
(Pmax 100, L* and W_brake, ``results/acc``), Car2D (``results/car2d``) and the
final Docking2D controllers (``results/docking2d/seed<s>``). Each run gets
``synthesis/`` beside its ``controller.pt``, which ``acc.py``, ``car2d.py`` and
``docking2d.py report`` read; a run whose ``synthesis.json`` is newer than its
controller is skipped unless ``--force``. ``--summarize`` only recomputes the
``synthesis.json`` of synthesized runs from their Vivado reports (no Vivado).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _runs import report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCHMARKS = ("acc", "car2d", "docking2d")


def runs(benchmark, results):
    """Run directories of the certified controllers whose patches the benchmark's report lists."""
    if benchmark == "acc":
        import acc
        candidates = [acc.output(results, method, 100, seed) for method in ("invariant", "full_braking")
                      for seed in acc.SEEDS]
    elif benchmark == "car2d":
        candidates = [results / "wall_box_b_1x1"]
    else:
        import docking2d
        candidates = [results / f"seed{seed}" for seed in docking2d.SEEDS]
    return [run for run in candidates if (report(run) or {}).get("status") == "safe"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmarks", nargs="*", default=list(BENCHMARKS), help=f"any of {', '.join(BENCHMARKS)} (default: all)")
    parser.add_argument("--results", type=Path, default=ROOT / "results",
                        help="holds acc/, car2d/ and docking2d/ as the experiment scripts write them")
    parser.add_argument("--jobs", type=int, default=4, help="Vivado runs at a time")
    parser.add_argument("--force", action="store_true", help="also redo runs that are already synthesized")
    parser.add_argument("--summarize", action="store_true",
                        help="only recompute synthesis.json from the existing Vivado reports")
    parser.add_argument("--vivado-settings", help="Vivado's settings64.sh (default: $VIVADO_SETTINGS)")
    parser.add_argument("--part", default=None, help="FPGA part (default: circuit.synthesize.PART)")
    parser.add_argument("--stages", type=int, default=None, help="the DWC's PIPE_STAGES (default: 3)")
    arguments = parser.parse_args()
    unknown = set(arguments.benchmarks) - set(BENCHMARKS)
    if unknown:
        parser.error(f"unknown benchmark(s) {', '.join(sorted(unknown))}; choose from {', '.join(BENCHMARKS)}")
    if arguments.vivado_settings:
        os.environ["VIVADO_SETTINGS"] = arguments.vivado_settings
    if not arguments.summarize and not os.environ.get("VIVADO_SETTINGS"):
        parser.error("set VIVADO_SETTINGS or pass --vivado-settings")

    from circuit.measure import SYNTHESIZED, synthesized_row
    from circuit.synthesize import PART, summarize, synthesize
    from reacheco.policy import DWCPolicy

    if arguments.summarize:
        done = [run for benchmark in dict.fromkeys(arguments.benchmarks)
                for run in runs(benchmark, arguments.results / benchmark)
                if (run / "synthesis" / "synthesis.json").exists()]
        _table([(run, summarize(run / "synthesis")) for run in done], SYNTHESIZED, synthesized_row)
        return

    todo, found = [], 0
    for benchmark in dict.fromkeys(arguments.benchmarks):
        for run in runs(benchmark, arguments.results / benchmark):
            found += 1
            done = run / "synthesis" / "synthesis.json"
            if arguments.force or not done.exists() or done.stat().st_mtime < (run / "controller.pt").stat().st_mtime:
                todo.append(run)
    print(f"{found} certified controller(s), {len(todo)} to synthesize", flush=True)

    # Load in this thread: torch.load in several threads at once deadlocks on its lazy imports.
    policies = {run: DWCPolicy.load(run / "controller.pt") for run in todo}

    def one(run):
        summary = synthesize(policies[run], run / "synthesis", part=arguments.part or PART, stages=arguments.stages)
        print(f"{run}: patch {int(summary['patch']['lut'])} LUT", flush=True)
        return run, summary

    with ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        finished = list(pool.map(one, todo))
    _table(finished, SYNTHESIZED, synthesized_row)


def _table(finished, columns, row):
    if finished:
        width = max(len(str(run)) for run, _ in finished)
        print("\n" + "run".ljust(width), *columns, sep="  ")
        for run, summary in finished:
            print(str(run).ljust(width), *row(summary), sep="  ")


if __name__ == "__main__":
    main()
