"""Record the prototype's policy outputs as regression data for ``tests/test_policy.py``.

Usage::

    python -m tools.make_policy_goldens PROTOTYPE_ROOT NAME=SOURCE.dwc.pt [...]

For every prototype artifact this stores, under ``tests/golden/policy/NAME.pt``,
the native value of every grid cell, the decoded action of every native value
in ``0 .. max``, and the upper repair bound. Only the prototype loads SOURCE;
the matching ReachECO checkpoint is produced by ``tools.convert_prototype_checkpoint``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "policy"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prototype", type=Path)
    parser.add_argument("sources", nargs="+", help="NAME=SOURCE pairs")
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.prototype.resolve()))
    from policies import load_policy

    commit = subprocess.run(["git", "-C", str(arguments.prototype), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for pair in arguments.sources:
        name, source = pair.split("=", 1)
        policy = load_policy(source, device="cpu")
        cells = torch.arange(int(policy.sizes.prod()))
        values = torch.arange(policy.max_repair_integer + 1, dtype=torch.float32)[:, None]
        golden = {
            "cell_values": policy.cell_repair_values(cells).to(torch.long),
            "decoded": policy.decode_actions(values.expand(-1, policy.action_dim)),
            "max_value": int(policy.max_repair_integer),
            "provenance": {"prototype_commit": commit, "source": str(Path(source).resolve())},
        }
        torch.save(golden, GOLDEN / f"{name}.pt")
        print(f"{name}: {len(cells)} cells, repair values 0..{golden['max_value']}")


if __name__ == "__main__":
    main()
