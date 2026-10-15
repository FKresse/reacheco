# ReachECO

Code for the paper *Verification-Guided Safety-Repair of Learned Controllers*
(Fabian Kresse, Sobhan Gharanfoli, Nikolaus Vertovec, Alessandro Abate, Thomas A. Henzinger, Christoph H. Lampert).

ArXiv Link will follow soon.

ReachECO certifies a Differentiable Weightless Controller (DWC) with forward
reachability and, where it fails, repairs it backward: the fix is an exact
per-cell patch that can be synthesized next to the controller on an FPGA.

## Installation

Python ≥ 3.9 and a CUDA GPU (CPU works, but slowly). The paper's runs used Python 3.9
and torch 2.3.0 with CUDA 11.8; install that torch first:

```bash
pip install torch==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .                  # numpy, gymnasium, ortools
```

The baselines also need [torch-dwn](https://github.com/alanbacellar/DWN), whose CUDA
extension is built from source against that torch (with CUDA 11.8 and gcc 10; newer gcc
may fail):

```bash
module load cuda/11.8 gcc/10      # on a cluster; otherwise install them locally
git clone https://github.com/alanbacellar/DWN
pip install setuptools wheel
pip install -v --no-build-isolation ./DWN/
pip install -e ".[baselines]"     # + stable-baselines3 2.3.2, matplotlib
```

## Layout

```
reacheco/      the library
checkpoints/   the original controllers (acc, docking2d, car2d)
problems/      verification problems (initial set, unsafe set, goal)
backends/      continuation backend config files (the paper's W)
scripts/       experiments of the paper
baselines/     baselines of the paper
circuit/       patch → Verilog → FPGA synthesis
results/       everything the scripts write
```

## Usage

Repair a single controller (with losing set learning enabled):

```bash
python scripts/repair.py --problem problems/acc/boundary160_collision_30.json \
    --checkpoint checkpoints/acc/dwc_b81_seed1.pt --backend backends/acc_game_11.json \
    --output results/example
```

The output directory holds the repaired `controller.pt`, `report.json` and `run.json`
(command line, commit, GPU). The repaired controller loads like the original:

```python
from reacheco.policy import DWCPolicy

policy = DWCPolicy.load("results/example/controller.pt", device="cuda")
actions = policy.decode(policy.cell_values(policy.cell_ids(states)))   # states: (N, D)
```

## Reproducing the experiments

Run everything from the repository root. Each run gets its own directory under
`results/`; interrupted sweeps resume where they stopped.

```bash
# ACC (Tables 1, 2, Figure 6): 5 methods x Pmax 10/20/30/100 x 5 seeds, 1 h each
python scripts/acc.py run --gpus 0 1 2
python scripts/acc.py report            # -> results/acc/tables.md

# Docking2D (Table 3, Figure 3)
python scripts/docking2d.py run --seeds 1 2 3 --gpus 0 1 2
python scripts/docking2d.py report      # -> results/docking2d/report.md

# Car2D (Appendix A.2)
python scripts/car2d.py run
python scripts/car2d.py report          # -> results/car2d/report.md
```

`--gpus`, `--seeds`, `--methods` and `--pmax` select a subset; see `--help`.

### Rollout evaluation

Rolls out the original and the repaired controllers from the same 1,000 starts
(ACC Table 5, Docking2D Table 3, Car2D); the `report` commands pick the results up.

```bash
python scripts/evaluate.py acc
python scripts/evaluate.py car2d
python scripts/evaluate.py docking2d
```

Docking2D runs in the [SafeRL](https://github.com/act3-ace/SafeRL) environment
the controllers were trained in. It needs old versions (ray 1.5.2, gym 0.22,
Python 3.9), so it is not installed with ReachECO:

```bash
git clone https://github.com/act3-ace/SafeRL
pip install --no-deps -e SafeRL     # its requirements.txt pins far more than the environment needs
pip install "ray[rllib]==1.5.2" "gym==0.22.0" "numpy<2" "protobuf<3.21" scipy pyyaml jsonlines
```

Note regarding Docking2D: SafeRL uses a slightly stricter safety condition then we (and the AINNCS competition) verify. Because of that, for seed 3, we can observed two unsafe transitions in SafeRL which are not present under our verified and repaired safety specification.

## Baselines

```bash
python baselines/supervisedECO/run.py run --gpus 0   # BBReach + Supervised
python baselines/stlgym/run.py run                   # STLGym (CPU only)
```

Both have a `report` command like the scripts above.

## Hardware synthesis

The controllers are [DWCs](https://github.com/FKresse/differentiable_weightless_controllers),
and `circuit/` reuses their Verilog. It compiles each patch into a separate block
behind the controller and synthesizes both with Vivado (tested with 2022.2,
xc7a100t, 100 MHz):

```bash
export VIVADO_SETTINGS=/path/to/Xilinx/Vivado/2022.2/settings64.sh
python scripts/synthesize.py                # acc, car2d and docking2d; or name some
```

Results go to `synthesis/` next to each repaired controller and show up in the reports.
Patch sizes without Vivado (the LUT6 estimate): `python -m circuit.measure <runs>`.