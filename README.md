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

## Layout

```
reacheco/        the library
checkpoints/acc/ ACC DWC controllers: dwc_b81_seed{1..5} (paper), dwc_b159 (regression)
tools/           one-off conversion and golden-data scripts (need the prototype checkout)
tests/           pytest suite; tests/golden holds prototype outputs at commit 9fb2fbef
```

## Usage

```python
from reacheco.policy import DWCPolicy

policy = DWCPolicy.load("checkpoints/acc/dwc_b81_seed1.pt", device="cuda")
values = policy.box_values(boxes)   # native values of policy-aligned boxes (N, 2, D)
actions = policy.decode(values)     # normalized actions
```

## Tests

```bash
python -m pytest
```
