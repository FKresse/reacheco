"""Run many script jobs (each writing one output directory) on several GPUs and collect their reports."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPAIR = Path(__file__).resolve().parent / "repair.py"


def run_jobs(jobs, gpus, *, timeout=3600, per_gpu=1):
    """Run ``(output, script, arguments)`` jobs, ``per_gpu`` at a time on each GPU.

    Each job runs ``python script *arguments --output output``.
    A job whose output already holds ``report.json`` or ``TIMEOUT`` is
    skipped, so an interrupted sweep resumes. A job running longer than
    ``timeout`` seconds is killed and leaves a ``TIMEOUT`` marker.
    """
    pending = [(Path(output), script, arguments) for output, script, arguments in jobs
               if not (Path(output) / "report.json").exists() and not (Path(output) / "TIMEOUT").exists()]
    print(f"{len(pending)} of {len(jobs)} runs to do on GPUs {list(gpus)}", flush=True)
    slots = queue.Queue()
    for gpu in gpus:
        for _ in range(per_gpu):
            slots.put(str(gpu))

    def run(job):
        output, script, arguments = job
        gpu = slots.get()
        try:
            output.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            with (output / "run.log").open("w") as log:
                try:
                    subprocess.run([sys.executable, str(script), *map(str, arguments), "--output", str(output)],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
                    outcome = report(output)
                    outcome = outcome["status"] if outcome else "error (see run.log)"
                except subprocess.TimeoutExpired:
                    (output / "TIMEOUT").touch()
                    outcome = "timeout"
            print(f"{output}: {outcome} in {time.monotonic() - started:.0f} s (GPU {gpu})", flush=True)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus) * per_gpu) as pool:
        list(pool.map(run, pending))


def report(output):
    """The run's ``report.json``, ``{"status": "timeout"}`` for a timed-out run, or ``None``."""
    output = Path(output)
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    if (output / "TIMEOUT").exists():
        return {"status": "timeout"}
    return None


def patch_lut6(output):
    """Patch size in LUT6 of the run's ``controller.pt`` (``circuit``), cached in ``patch_circuit.json``."""
    controller, cached = Path(output) / "controller.pt", Path(output) / "patch_circuit.json"
    if cached.exists() and cached.stat().st_mtime >= controller.stat().st_mtime:
        return json.loads(cached.read_text())["total_lut6"]
    from circuit.measure import measure
    return measure(controller)["total_lut6"]


def synthesized_patch(output):
    """Synthesized patch LUTs of the run (``scripts/synthesize.py``) as ``(logic, memory)``, or ``None``.

    ``logic`` (guards, address encoder, adders; LUT as Logic, without the pipeline's
    shift-register LUTs) compares with the LUT6 estimate, ``memory`` is the correction memory's block.
    """
    path = Path(output) / "synthesis" / "synthesis.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if "memory" not in summary:
        raise ValueError(f"{path} predates the separate memory block; rerun scripts/synthesize.py --force")
    if "lut_logic" not in summary["logic"]:
        raise ValueError(f"{path} predates the shift-register split; run scripts/synthesize.py --summarize")
    return int(summary["logic"]["lut_logic"]), int(summary["memory"]["lut"])


def evaluation(output):
    """The run's rollout evaluation (``scripts/evaluate.py``), or ``None``."""
    path = Path(output) / "evaluation.json"
    return json.loads(path.read_text()) if path.exists() else None


def rollouts(result, digits=2):
    """``(reward mean ± std, unsafe [%])`` of one evaluated controller, or dashes."""
    if result is None:
        return "–", "–"
    return f"{result['reward_mean']:.{digits}f} ± {result['reward_std']:.{digits}f}", f"{result['unsafe_percent']:.1f}"


def mean_std(values, digits=0):
    """``"mean ± std"`` (sample standard deviation), or ``"–"`` without values."""
    if not values:
        return "–"
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)
