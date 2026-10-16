# Docking2D (box 70..120, four velocity quadrants)

| seed | status | passes | repairs | changed cells | t [s] | patch logic [LUT] | patch memory [LUT] | patch [LUT] |
|---|---|---|---|---|---|---|---|---|
| 1 | safe | 2 | 4 | 5 | 904 | 36 | 14 | 50 |
| 2 | safe | 1 | 0 | 0 | 1352 | 0 | 0 | 0 |
| 3 | safe | 3 | 5259 | 3309 | 3924 | 5249 | 1166 | 6415 |

## Table 3: rollouts in SafeRL (scripts/evaluate.py docking2d)

| seed | safe [%] X | safe [%] X0 | safe [%] X after | reward X0 | reward X0 after |
|---|---|---|---|---|---|
| 1 | 100.0 | 100.0 | 100.0 | 3.066 ± 0.067 | 3.066 ± 0.067 |
| 2 | 95.6 | 100.0 | 95.6 | 2.994 ± 0.061 | 2.994 ± 0.061 |
| 3 | 96.0 | 75.9 | 98.5 | 3.003 ± 0.148 | 3.060 ± 0.066 |
