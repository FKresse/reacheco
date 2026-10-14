# ReachECO

Verification-guided safety repair of learned controllers: forward closed-loop
reachability with lazy backward repair, realized as exact per-cell patches of
Differentiable Weightless Controllers (DWCs).

This repository is a clean, audited port of the research prototype
(`release_closed_loop`, commit `9fb2fbef`). It is built step by step, each
module checked against the prototype on the ACC benchmark.

## Layout

```
reacheco/               the library
checkpoints/            original DWC controllers (acc, docking2d, car2d)
backends/               continuation backends of the paper's runs (FindCont's W)
problems/               verification problems: ACC (X0 cut at p <= 10, 20, 30, 100),
                        Docking2D (X0 and its four velocity quadrants), Car2D (box B); problems/acc/sets/
                        also holds the paper's initial losing set
```

Tests, golden data and the conversion tools live outside this repository in
`ReachECO-dev`; the golden data are prototype outputs at commit `9fb2fbef`.

## Reproducing the experiments

Scripts in `scripts/` run from the repository root and write to `results/`
(not versioned). Each run is one process with its own directory holding the
repaired controller, the merge mask, the losing set, `report.json` and
`run.json` (command line, commit, GPU); finished or timed-out runs are
skipped when a sweep is resumed.

```
python scripts/repair.py --problem ... --checkpoint ... --backend ... --output DIR   # one repair
python scripts/acc.py run --gpus 1 2 3        # ACC: 5 methods x Pmax 10/20/30/100 x seeds 1-5, 1 h each
python scripts/acc.py tables                  # Tables 1, 2 and Figure 6 data -> results/acc/tables.md
```

Not yet available: Docking2D and Car2D sweeps, rollout evaluation, and patch
sizes in LUT6s.

## Usage

```python
from reacheco.policy import DWCPolicy

policy = DWCPolicy.load("checkpoints/acc/dwc_b81_seed1.pt", device="cuda")
cells = policy.cell_ids(states)             # policy cells of states (N, D)
actions = policy.decode(policy.cell_values(cells))   # their native values, decoded to actions
```
