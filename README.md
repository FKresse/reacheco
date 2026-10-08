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
problems/               verification problems: ACC (X0 cut at p <= 10, 20, 30, 100),
                        Docking2D (X0 and its four velocity quadrants), Car2D (box B)
```

Tests, golden data and the conversion tools live outside this repository in
`ReachECO-dev`; the golden data are prototype outputs at commit `9fb2fbef`.

## Usage

```python
from reacheco.policy import DWCPolicy

policy = DWCPolicy.load("checkpoints/acc/dwc_b81_seed1.pt", device="cuda")
values = policy.box_values(boxes)   # native values of policy-aligned boxes (N, 2, D)
actions = policy.decode(values)     # normalized actions
```
