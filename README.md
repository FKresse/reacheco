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

## Known limitations

- **No outward rounding.** `Post` and `Tube` bounds are computed in float32
  without rounding outward, so a bound can be off by about one float32 step
  (up to 5e-7 observed for ACC). The paper also leaves floating-point errors
  out of scope. To be fixed later, e.g. by `nextafter` on the computed bounds.

## Layout

```
reacheco/        the library
checkpoints/acc/ ACC DWC controllers: dwc_b81_seed{1..5} (paper), dwc_b159 (regression)
problems/acc/    verification problems of the paper (X0 cut at p <= 10, 20, 30, 100)
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
