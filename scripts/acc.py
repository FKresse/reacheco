"""ACC experiments: Tables 1 and 2 and the data of Figure 6.

Run from the repository root::

    python scripts/acc.py run --gpus 1 2 3                   # all methods, Pmax 10 20 30 (100), seeds 1-5
    python scripts/acc.py run --methods invariant --pmax 30 --seeds 1
    python scripts/acc.py report                              # results/acc/tables.md

Each run repairs ``checkpoints/acc/dwc_b81_seed<s>.pt`` from the boundary-hugging
initial set cut at ``p <= Pmax`` with one of five methods (the paper's rows):

    no_losing_set       collision, 11-value alphabet, L = U throughout
    online_losing_set   collision, alphabet, L learned during repair
    initial_losing_set  as online, starting from the precomputed losing set
    full_braking        collision, full-braking backup (W_brake), L learned
    invariant           braking invariant as the unsafe set (L*), alphabet

and writes ``results/acc/<method>/p<Pmax>/seed<s>/``. At Pmax 100 only the
methods of Table 2 run (``PMAX_100``); the others time out there. Runs are
limited to one hour, as in the paper. ``report`` reads the reports: Table 1 (runtime at
Pmax 30), Table 2 (Pmax 100: certified seeds, runtime, action change), the
game depths of Figure 6 (Pmax 30) and an overview of all runs. Patch sizes
(LUT6, ``circuit``) are cached in each run's ``patch_circuit.json``; the
synthesized size is shown where ``python -m circuit.measure <runs> --synthesize``
was run (``synthesis/synthesis.json``).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from _runs import REPAIR, markdown, mean_std, patch_lut6, report, run_jobs, synthesized_patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = {
    # name: (unsafe set, backend, learn L, start from the initial losing set)
    "no_losing_set": ("collision", "acc_game_11.json", False, False),
    "online_losing_set": ("collision", "acc_game_11.json", True, False),
    "initial_losing_set": ("collision", "acc_game_11.json", True, True),
    "full_braking": ("collision", "acc_full_braking.json", True, False),
    "invariant": ("invariant", "acc_game_11.json", False, False),
}
LABELS = {"no_losing_set": "No losing set", "online_losing_set": "Online losing set",
          "initial_losing_set": "Initial losing set", "full_braking": "W_brake", "invariant": "L*"}
PMAX = (10, 20, 30, 100)
PMAX_100 = ("full_braking", "invariant")
SEEDS = (1, 2, 3, 4, 5)


def output(results, method, pmax, seed):
    return results / method / f"p{pmax}" / f"seed{seed}"


def jobs(results, methods, pmaxes, seeds):
    for pmax in pmaxes:           # the cheap sizes first, so a cut-short sweep still fills columns
        for method in methods:
            if pmax == 100 and method not in PMAX_100:
                continue
            unsafe, backend, learn, initial = METHODS[method]
            for seed in seeds:
                arguments = ["--problem", ROOT / f"problems/acc/boundary160_{unsafe}_{pmax}.json",
                             "--checkpoint", ROOT / f"checkpoints/acc/dwc_b81_seed{seed}.pt",
                             "--backend", ROOT / "backends" / backend]
                if initial:
                    arguments += ["--initial-losing-set", ROOT / "problems/acc/sets/initial_losing_set.json"]
                if not learn:
                    arguments.append("--no-learn")
                yield output(results, method, pmax, seed), REPAIR, arguments


@torch.no_grad()
def closing_box(pmax, *, braking=100.0, samples=100_000, seed=0):
    """``samples`` uniform states of the closing box ``0 <= p <= Pmax, -sqrt(2 B Pmax) <= v <= 0`` (fixed seed)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((samples, 2), generator=generator) * torch.tensor([pmax, -math.sqrt(2.0 * braking * pmax)])


def action_change(original, repaired, pmax, *, braking=100.0, samples=100_000, seed=0):
    """Change [%] and mean |delta| of the action on the closing box ``0 <= p <= Pmax, -sqrt(2 B Pmax) <= v < 0``.

    The box contains the states below the braking boundary too, where many
    repairs lie; the change per state is the largest over action components.
    
    The reported results in the paper for DWC are not monte carlo but exact.
    """
    states = closing_box(pmax, braking=braking, samples=samples, seed=seed).to(original.device)

    def act(policy):
        return policy.decode(policy.cell_values(policy.cell_ids(states)))

    delta = (act(repaired) - act(original)).abs().amax(1)
    # count if change is larger than 10^-6
    return 100.0 * float((delta > 1e-6).float().mean()), float(delta.mean())


def tables(results, seeds):
    from reacheco.policy import DWCPolicy

    def reports(method, pmax):
        return [report(output(results, method, pmax, seed)) for seed in seeds]

    def certified(method, pmax):
        return [r for r in reports(method, pmax) if r and r["status"] == "safe"]

    def count(method, pmax):
        done = [r for r in reports(method, pmax) if r]
        return f"{len(certified(method, pmax))}/{len(done)}" + ("" if len(done) == len(seeds) else f" ({len(seeds)} seeds)")

    sections = []
    rows = [(LABELS[m], mean_std([r["seconds"] for r in certified(m, 30)]), count(m, 30)) for m in METHODS]
    sections.append("## Table 1: runtime at Pmax = 30 (certified runs)\n\n"
                    + markdown(("Method", "t [s]", "certified"), rows))

    rows = []
    for method in ("invariant", "full_braking"):
        changes, deltas, sizes, logic, memory = [], [], [], [], []
        for seed in seeds:
            run = output(results, method, 100, seed)
            if (report(run) or {}).get("status") != "safe":
                continue
            original = DWCPolicy.load(ROOT / f"checkpoints/acc/dwc_b81_seed{seed}.pt")
            change, delta = action_change(original, DWCPolicy.load(run / "controller.pt"), 100)
            changes.append(change)
            deltas.append(delta)
            sizes.append(patch_lut6(run))
            synthesized = synthesized_patch(run)
            if synthesized is not None:
                logic.append(synthesized[0])
                memory.append(synthesized[1])
        rows.append((LABELS[method], count(method, 100), mean_std([r["seconds"] for r in certified(method, 100)]),
                     mean_std(sizes), mean_std(logic), mean_std(memory), mean_std(changes, 1), mean_std(deltas, 3)))
    sections.append("## Table 2: Pmax = 100\n\n"
                    + markdown(("Method", "certified", "t [s]", "Param. [LUT6]", "synth. logic [LUT]",
                                "synth. memory [LUT]", "Change [%]",
                                "mean \\|Δ\\|"), rows))

    depths = {}
    for method in METHODS:
        for r in certified(method, 30):
            counts = {}
            for entry in r["repairs"]:
                for key, value in entry.items():
                    if key.startswith("horizon_"):
                        counts[int(key[8:])] = counts.get(int(key[8:]), 0) + value
            depths.setdefault(method, []).append(counts)
    deepest = max((d for runs in depths.values() for counts in runs for d in counts), default=0)
    rows = [(LABELS[m], *(mean_std([counts.get(d, 0) for counts in depths[m]], 1) for d in range(1, deepest + 1)))
            for m in METHODS if m in depths]
    sections.append("## Figure 6: games per depth at Pmax = 30 (mean ± std over certified seeds)\n\n"
                    + markdown(("Method", *map(str, range(1, deepest + 1))), rows))

    rows = []
    for method in METHODS:
        for pmax in PMAX:
            done = [r for r in reports(method, pmax) if r]
            if not done:
                continue
            safe = certified(method, pmax)
            rows.append((LABELS[method], pmax, count(method, pmax), mean_std([r["seconds"] for r in safe]),
                         mean_std([len(r["repairs"]) for r in safe], 1), mean_std([r["overrides"] for r in safe], 1),
                         sum(r["status"] == "timeout" for r in done)))
    sections.append("## All runs\n\n" + markdown(("Method", "Pmax", "certified", "t [s]", "repairs",
                                                  "changed cells", "timeouts"), rows))
    path = results / "tables.md"
    path.write_text("# ACC\n\n" + "\n\n".join(sections) + "\n")
    print(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--results", type=Path, default=ROOT / "results/acc")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--pmax", nargs="+", type=int, default=list(PMAX))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--per-gpu", type=int, default=1, help="concurrent runs per GPU")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    arguments = parser.parse_args()
    if arguments.command == "run":
        run_jobs(list(jobs(arguments.results, arguments.methods, arguments.pmax, arguments.seeds)),
                 arguments.gpus, timeout=arguments.timeout, per_gpu=arguments.per_gpu)
    else:
        tables(arguments.results, arguments.seeds)


if __name__ == "__main__":
    main()
