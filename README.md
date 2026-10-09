# ReachECO

Verification-guided safety repair of learned controllers: forward closed-loop
reachability with lazy backward repair, realized as exact per-cell patches of
Differentiable Weightless Controllers (DWCs).

This repository is a clean, audited port of the research prototype
(`release_closed_loop`, commit `9fb2fbef`). It is built step by step, each
module checked against the prototype on the ACC benchmark.

## Status

| Component | Module | State |
|---|---|---|
| DWC policy loading, cell values, decoding, overrides | `reacheco/policy.py` | ported |
| Box splitting along the policy grid, goal cutting | `reacheco/boxes.py` | ported |
| Safety specifications, problem files | `reacheco/spec.py` | ported |
| ACC: interval propagators, training and evaluation rollouts | `reacheco/environments/acc.py` | ported |
| Docking2D: CWH propagators (rollouts: SafeRL, not included) | `reacheco/environments/docking2d.py` | ported |
| Car2D: exact-flow propagators, course rollouts | `reacheco/environments/car2d.py` | ported |
| Intersample tube certification (Tube part of SafeStep) | `reacheco/tubes.py` | ported |
| ReachStep: forward step with per-cell merging, fixed-point test | `reacheco/reach.py` | ported |
| Forward verification without repair | `reacheco/verify.py` | ported |
| Losing set `L` (certified losing boxes per policy cell) | `reacheco/losing.py` | ported |
| FindCont: continuation games, repair constraints, backends (alphabet, Car2D steering corrections, constant and Docking2D backups) | `reacheco/continuation.py` | ported |
| Backtrack: obligations to parent boxes, lazy unmerging replay, cone merge protection | `reacheco/backtrack.py` | ported |
| Repair constraints: per-cell compression of one-step constraints | `reacheco/constraints.py` | ported |
| Joint: CP-SAT synthesis of one controller, conflict cores | `reacheco/joint.py` | ported |
| BRepair and ReachECO (Algorithms 1 and 2), EarliestAffected, pruning | `reacheco/repair.py` | ported |

## Deviations from the prototype

Behaviour that intentionally differs from the research prototype:

- DWC repair values may also lie below 0, so that the repair alphabet reaches
  both action clips (`DWCPolicy.min_value`).
- Boxes with NaN or inverted bounds are rejected instead of silently dropped.
- The ACC braking-invariant specification takes its braking acceleration from
  the system configuration.
- `ACC.step` reports only `crash` and `too_far` in `info`.
- `--no-merge` keeps every successor piece as its own row directly, instead of
  marking every policy cell merge-disabled; the forward pass is identical.
- Forward-step records list every policy cell whose action was read
  (`touched_cells`), including cells read by the merge lookahead.
- Merging on grid dimensions with a single cell (used only by the UUV case
  study) is not supported.
- The DWC decoder computes `exp(log_alpha)` on the CPU, so actions do not
  depend on the device. The prototype computed it on the device; on CUDA this
  differs by one float32 step for some controllers (e.g. Docking2D seed 1,
  Car2D, ACC b81 seed 3).
- Docking2D propagators do not treat boxes inside the goal specially (the
  prototype returned them unchanged); the goal is cut off before propagation.
- The intersample tube schedule is part of the problem file
  (`tube_segments`), not a global setting.
- Losing-set cell boxes come from the policy thresholds (the prototype assumed
  a uniform grid, which is wrong for Docking2D's Gaussian thresholds; ACC is
  unchanged). An upper outer cell lying entirely beyond the grid bound has no
  cell box and is never certified as a whole.
- The losing set tabulates only occupied cells on the device (the prototype
  allocated per-cell tables over the whole grid, about 11 GB for Docking2D).
- When certifying a whole policy cell as losing, each successor piece is
  checked against `L` in its own cell. The prototype paired the pieces after
  the goal cut with the cells from before it, which the cut can reorder (no
  effect on the recorded cases).
- The Car2D continuation backend (`LocalAlphabet`) proposes corrections of the
  network's own value of each cell, the same at every game depth. The
  prototype corrected the current, possibly already repaired, value and added
  values other constraints allowed on the cell ("seeds"); its proposals moved
  with every repair, so it could not learn `L`. With fixed proposals per cell
  it learns `L` and certifies whole cells like the alphabet, and games of
  different obligations offer the same values on a cell without seeds.
- Backtracking always protects the cells of the failure's ancestor cone from
  merging; the prototype's older rule (every cell its replay visited) and its
  conditional mode (constraints conditioned on the decision prefix) are not
  ported.
- EarliestAffected restarts at the first step that read a changed cell,
  including the cells read by the merge lookahead (`touched_cells`); the
  prototype only looked at the cells of the source boxes.
- When the losing set grows, the forward pass resumes at the first step that
  queried `L` with a box meeting the added boxes (its rows and premerge
  pieces, one step earlier for the merge lookahead), not at `X_0`. The result
  is identical; Car2D restarts later in 3 of 14 repairs (about 5% faster),
  ACC almost always still at `X_0`.
- Repairs no longer read by the certifying pass are reverted at the end of the
  run (pruning), using the cells that pass read; the prototype pruned in a
  separate pass without the losing set, the time-specific merge mask and the
  merge lookahead. The saved merge mask is time-specific, so the certifying
  pass can be reproduced exactly.
- Joint synthesis always rebuilds its model; the prototype's incremental mode
  (used for Car2D in the paper) gives the same controller.
- Every backend certifies whole losing cells, with the values it offers on the
  cell. The prototype did this only for alphabets; for backups (ACC full
  braking, Docking2D approach) it stored only the losing boxes themselves.

## Known limitations

- **No outward rounding for ACC and Car2D.** Their `Post` and `Tube` bounds
  are computed in float32 without rounding outward, so a bound can be off by
  about one float32 step (up to 5e-7 observed for ACC). Docking2D computes in
  float64 and rounds outward. The paper leaves floating-point errors out of
  scope. To be fixed later, e.g. by `nextafter` on the computed bounds.

## Layout

```
reacheco/               the library
checkpoints/            original DWC controllers (acc, docking2d, car2d)
checkpoints_repaired/   repaired controllers of the paper's runs, each with the merge
                        mask(s) of its certifying pass and source.json (provenance)
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
values = policy.box_values(boxes)   # native values of policy-aligned boxes (N, 2, D)
actions = policy.decode(values)     # normalized actions
```
