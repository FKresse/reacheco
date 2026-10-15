"""Synthesize a repaired controller with and without its patch in Vivado, after simulating both.

Needs Vivado (2022.2 tested): ``VIVADO_SETTINGS`` names its ``settings64.sh``.
By default the designs read signed 12-bit observations through the threshold
comparators (``hdl.encoder``, grid bounds mapped onto the signed range).
For each design (``base``: the controller; ``patched``: controller and patch,
``hdl.top``), in ``<directory>/<design>/``:

1. With ``check``, ``xsim`` drives every grid cell (or every repaired cell
   plus a sample, as ``patch.verify_patch``) through the design and requires
   the policy's native values (the network's for ``base``, the repaired ones
   for ``patched``) bit for bit.
2. Vivado synthesizes, places and routes ``top`` out of context on ``PART``
   with a ``period`` ns clock and reports utilization (for ``patched`` also of
   the patch instance alone) and timing.

Both designs use the DWC repository's ``PIPE_STAGES`` pipeline (``stages``,
default ``hdl.default_stages``). The patch is a separate block
(``keep_hierarchy``): ``patch`` is its own utilization, split into its
correction ``memory`` (a block of its own, ``patch_memory``) and the rest,
``logic`` (guards, address encoder, adders, pipeline); ``difference``
(``patched - base``) is kept for reference. ``estimate`` holds the patch's
LUT6 estimate without the memory (``patch.compile_patch``, comparable with
``logic``) and the memory as a ROM in LUTs (``cost.rom_luts``, comparable with
``memory``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

from . import cost, hdl
from .patch import cell_clusters, cell_coordinates, compile_patch, grid_code

PART = "xc7a100tfgg484-1"
WORD = 12
SAMPLE = 20_000
METRICS = {"lut": r"Slice LUTs\*?", "ff": r"Slice Registers", "carry": r"CARRY4", "lutram": r"LUT as Memory",
           "muxf7": r"F7 Muxes", "bram": r"Block RAM Tile"}

SCRIPT = """read_verilog design.v
synth_design -top top -part {part} -mode out_of_context
create_clock -name clk -period {period} [get_ports clk]
opt_design
place_design
route_design
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
set slack [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set f [open slack.txt w]; puts $f $slack; close $f
{patch_report}
"""


def synthesize(policy, directory, *, part=PART, period=10.0, word=WORD, stages=None, check=False, sample=SAMPLE,
               seed=0):
    """Implement (and with ``check`` first simulate) the controller with and without its patch; returns the summary.

    ``word``: signed input width of the thermometer encoder (``hdl.encoder``), ``None`` for thermometer-bit inputs.
    """
    if check and word:
        raise ValueError("the simulation check drives thermometer bits: use word=None")
    directory = Path(directory)
    stages = hdl.default_stages(policy) if stages is None else stages
    patch = cell_clusters(policy)
    # generate cell ids for testing
    ids = _cells(policy, sample, seed)
    bits = grid_code(policy).thermometer(cell_coordinates(policy, ids))
    summary = {"part": part, "period_ns": period, "input_word_bits": word, "pipe_stages": stages,
               "latency_cycles": hdl.latency(stages) + 3, "cells_simulated": len(ids) if check else 0}
    summary["estimate"] = estimate(policy)
    for name, values, design_patch in (("base", policy.network_values(ids), None),
                                       ("patched", policy.cell_values(ids), patch if len(patch) else None)):
        work = directory / name
        shutil.rmtree(work, ignore_errors=True)     # no leftovers of an earlier run
        work.mkdir(parents=True)
        design, out = hdl.top(policy, design_patch, word=word, stages=stages)
        (work / "design.v").write_text(design)
        simulation = _simulate(work, bits, values.long(), out, hdl.latency(stages) + 1) if check else None
        summary[name] = {"simulation": simulation, **_implement(work, part, period, design_patch is not None)}
    zero = {key: 0.0 for key in METRICS}
    summary["patch"] = summary["patched"].get("patch_cells", zero)
    summary["memory"] = summary["patched"].get("memory_cells", zero)
    summary["logic"] = {key: summary["patch"][key] - summary["memory"][key] for key in METRICS}
    summary["difference"] = {key: summary["patched"][key] - summary["base"][key] for key in METRICS}
    (directory / "synthesis.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def estimate(policy):
    """The patch's LUT6 estimate without and with its correction memory as a ROM in LUTs."""
    summary = compile_patch(policy, verify=False)
    patch = cell_clusters(policy)
    if not len(patch):
        return {"lut6": 0, "memory_lut6": 0, "lut6_with_memory": 0}
    words = cost.build_words(patch.deltas)
    memory = sum(cost.rom_luts(words.shape[1], cost.minimum_signed_width(row)) for row in words.tolist())
    return {"lut6": summary["total_lut6"], "memory_lut6": memory, "lut6_with_memory": summary["total_lut6"] + memory}


def _cells(policy, sample, seed):
    """Every cell of a small grid; otherwise every repaired cell and a uniform sample."""
    if policy.num_cells <= max(sample, 50_000):
        return torch.arange(policy.num_cells)
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.randint(0, policy.num_cells, (sample,), generator=generator)
    return torch.unique(torch.cat((torch.tensor(sorted(policy.overrides), dtype=torch.long), drawn)))


def _vivado(work, command):
    settings = os.environ.get("VIVADO_SETTINGS")
    if not settings:
        raise RuntimeError("set VIVADO_SETTINGS to Vivado's settings64.sh")
    shell = (f'export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${{LIBRARY_PATH:+:$LIBRARY_PATH}}; '
             f'source "{settings}" > /dev/null && {command}')
    completed = subprocess.run(["bash", "-c", shell], cwd=work, capture_output=True, text=True)
    (work / "vivado.out").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Vivado failed in {work} (see vivado.out)")
    return completed.stdout


def _simulate(work, bits, values, out, delay):
    """Run the design in ``xsim`` on the given cells; raises on any mismatch."""
    stimulus, expected = hdl.vectors(bits, values, out)
    (work / "stimulus.mem").write_text(stimulus)
    (work / "expected.mem").write_text(expected)
    (work / "tb.v").write_text(hdl.testbench(len(bits), bits.shape[1], out * values.shape[1], delay))
    log = _vivado(work, 'xvlog design.v tb.v "$XILINX_VIVADO/data/verilog/src/glbl.v" && '
                        "xelab -L unisims_ver tb glbl -s sim && xsim sim -R")
    found = re.search(r"CHECKED (\d+) ERRORS (\d+)", log)
    if not found or int(found.group(2)):
        raise AssertionError(f"simulation of {work} disagrees with the policy (see vivado.out)")
    return {"cells": int(found.group(1)), "errors": 0}


def _implement(work, part, period, patched):
    """Out-of-context synthesis, placement and routing; utilization and timing."""
    patch_report = ("report_utilization -cells [get_cells u_patch] -file patch_utilization.rpt\n"
                    "report_utilization -cells [get_cells u_patch/u_memory] -file memory_utilization.rpt") if patched else ""
    (work / "implement.tcl").write_text(SCRIPT.format(part=part, period=period, patch_report=patch_report))
    _vivado(work, "vivado -mode batch -nojournal -log vivado.log -source implement.tcl")
    slack = float((work / "slack.txt").read_text())
    result = {**_utilization(work / "utilization.rpt"), "wns_ns": slack, "fmax_mhz": 1000.0 / (period - slack)}
    if patched:
        result["patch_cells"] = _utilization(work / "patch_utilization.rpt")
        result["memory_cells"] = _utilization(work / "memory_utilization.rpt")
    return result


def _utilization(report):
    text = report.read_text()
    found = {}
    for key, label in METRICS.items():
        match = re.search(r"\|\s*" + label + r"\s*\|\s*([0-9.]+)", text)
        found[key] = float(match.group(1)) if match else 0.0
    return found
