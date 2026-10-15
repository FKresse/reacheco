"""Measure the patch circuit of every repaired controller under the given paths.

Run from the repository root::

    python -m circuit.measure results/acc/full_braking/p100 results/docking2d

Each ``controller.pt`` found gets a ``patch_circuit.json`` beside it; a table
of all runs is printed. With ``--synthesize`` (Vivado, ``VIVADO_SETTINGS``) each
controller is also simulated and implemented with and without its patch
(``circuit.synthesize``), into ``synthesis/`` beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit.patch import compile_patch
from reacheco.policy import DWCPolicy

COLUMNS = ("clusters", "guard_lut6", "encoder_lut6", "adder_lut6", "total_lut6", "memory_bits", "percent", "verified")
SYNTHESIZED = ("base_lut", "logic_lut", "estimate_lut6", "memory_lut", "memory_estimate_lut6", "patch_srl", "patch_ff",
               "patch_carry", "base_mhz", "patched_mhz")


def measure(path, *, verify=True, label=None):
    """Compile, verify and record one controller's patch; returns the summary."""
    summary = compile_patch(DWCPolicy.load(path), verify=verify, label=label or str(path.parent))
    summary["controller"] = str(path)
    (path.parent / "patch_circuit.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def row(summary):
    verification = summary["verification"] or {}
    return [summary["clusters"], summary["guard_lut6"], summary["encoder_lut6"], summary["adder_lut6"],
            summary["total_lut6"], summary["memory_bits"], round(summary["percent_of_controller"], 1),
            verification.get("scope", "-")]


def synthesized_row(summary):
    """One row of the synthesis table (``circuit.synthesize`` summary); ``logic_lut`` excludes the
    pipeline's shift-register LUTs (``patch_srl``)."""
    base, patch, estimate = summary["base"], summary["patch"], summary["estimate"]
    return [int(base["lut"]), int(summary["logic"]["lut_logic"]), estimate["lut6"], int(summary["memory"]["lut"]),
            estimate["memory_lut6"], int(patch["srl"]), int(patch["ff"]), int(patch["carry"]),
            round(base["fmax_mhz"], 1), round(summary["patched"]["fmax_mhz"], 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="controllers, or directories to search")
    parser.add_argument("--controller-name", default="controller.pt")
    parser.add_argument("--no-verify", action="store_true", help="skip the decoder check (sizes are then unverified)")
    parser.add_argument("--synthesize", action="store_true", help="also simulate and implement in Vivado")
    parser.add_argument("--part", default=None, help="FPGA part for --synthesize (default: circuit.synthesize.PART)")
    parser.add_argument("--check", action="store_true",
                        help="--synthesize: simulate each design against the policy first (needs --word 0)")
    parser.add_argument("--stages", type=int, default=None,
                        help="--synthesize: the DWC's PIPE_STAGES (default: 3)")
    parser.add_argument("--word", type=int, default=None,
                        help="--synthesize: signed input width of the encoder (default 12; 0: thermometer-bit inputs)")
    arguments = parser.parse_args()
    controllers = [c for p in arguments.paths for c in ([p] if p.is_file() else sorted(p.rglob(arguments.controller_name)))]
    if not controllers:
        raise SystemExit("no controllers found")
    rows = [(str(c.parent), row(measure(c, verify=not arguments.no_verify))) for c in controllers]
    width = max(len(name) for name, _ in rows)
    print("run".ljust(width), *COLUMNS, sep="  ")
    for name, values in rows:
        print(name.ljust(width), *values, sep="  ")
    if arguments.synthesize:
        from circuit.synthesize import PART, WORD, synthesize
        print("\nrun".ljust(width + 1), *SYNTHESIZED, sep="  ")
        for c in controllers:
            word = WORD if arguments.word is None else arguments.word or None
            summary = synthesize(DWCPolicy.load(c), c.parent / "synthesis", part=arguments.part or PART,
                                 word=word, stages=arguments.stages, check=arguments.check)
            print(str(c.parent).ljust(width), *synthesized_row(summary), sep="  ")


if __name__ == "__main__":
    main()
