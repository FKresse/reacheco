"""Convert a DWC artifact of the research prototype to the ReachECO format.

Usage::

    python -m tools.convert_prototype_checkpoint SOURCE.dwc.pt DESTINATION.pt

Accepted sources are ``closed-loop-dwc-policy`` (versions 1 and 2) and
``closed-loop-sparse-dwc-policy`` (versions 1 and 2) that observe every state
dimension and use no constant gates; this covers all ACC, Docking2D and Car2D
controllers. The prototype is not imported.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from reacheco.policy import FORMAT, VERSION, DWCPolicy


def convert(source):
    source = Path(source)
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    kind, version = artifact.get("format"), artifact.get("version")
    grid, network = artifact["grid"], artifact["network"]

    if kind == "closed-loop-dwc-policy" and version in (1, 2):
        lower, upper = grid["observation_min"], grid["observation_max"]
        rows = list(grid["thresholds"])
        overrides = artifact["overrides"]
        ids, values = overrides["cell_ids"], overrides["actions"]
    elif kind == "closed-loop-sparse-dwc-policy" and version in (1, 2):
        lower, upper = grid["lower"], grid["upper"]
        rows = list(grid["thresholds"])
        if list(grid.get("observed_dims", range(len(lower)))) != list(range(len(lower))):
            raise ValueError("partially observed grids are not supported")
        if grid.get("merge_thresholds"):
            raise ValueError("merge thresholds are not supported")
        ids = torch.tensor(sorted(artifact["overrides"]), dtype=torch.long)
        values = torch.tensor([artifact["overrides"][key] for key in ids.tolist()])
    else:
        raise ValueError(f"unsupported prototype artifact {kind!r} version {version!r}")
    if grid.get("comparison", "x > threshold") != "x > threshold":
        raise ValueError("unsupported thermometer comparison")
    if grid.get("flattening", "first-dimension-fast") != "first-dimension-fast":
        raise ValueError("unsupported cell flattening")
    if int(network.get("constant_count", 0)):
        raise ValueError("constant gates are not supported")

    action_dim = int(torch.as_tensor(network["beta"]).numel())
    converted = {
        "format": FORMAT,
        "version": VERSION,
        "grid": {
            "thresholds": [torch.as_tensor(row, dtype=torch.float32).reshape(-1).clone() for row in rows],
            "lower": torch.as_tensor(lower, dtype=torch.float32).reshape(-1).clone(),
            "upper": torch.as_tensor(upper, dtype=torch.float32).reshape(-1).clone(),
        },
        "network": {
            "mappings": [torch.as_tensor(m, dtype=torch.long).clone() for m in network["mappings"]],
            "luts": [torch.as_tensor(lut, dtype=torch.bool).clone() for lut in network["luts"]],
            "log_alpha": torch.as_tensor(network["log_alpha"], dtype=torch.float32).reshape(-1).clone(),
            "beta": torch.as_tensor(network["beta"], dtype=torch.float32).reshape(-1).clone(),
            "output_tanh": bool(network.get("output_tanh", False)),
            "action_clip": [float(v) for v in network.get("action_clip", (-1.0, 1.0))],
        },
        "overrides": {
            "cell_ids": torch.as_tensor(ids, dtype=torch.long).reshape(-1),
            "values": torch.as_tensor(values, dtype=torch.long).reshape(-1, action_dim),
        },
        "metadata": {
            **artifact.get("metadata", {}),
            "converted_from": {
                "path": str(source.resolve()),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "format": kind,
                "version": version,
            },
        },
    }

    policy = DWCPolicy(converted)
    domain = artifact.get("repair_domain")
    if domain is not None and (domain["max_integer"], domain["normalization_count"]) != (
        policy.max_value, policy.output_count
    ):
        raise ValueError(f"repair domain {domain} disagrees with the decoder")
    patch = artifact.get("repair_patch")
    if patch is not None and len(patch["cell_ids"]):
        # The prototype measured repair deltas from this reference; ReachECO
        # measures them from the network. Both must coincide.
        reference = torch.as_tensor(patch["reference"]).float().reshape(-1, action_dim)
        if not torch.equal(reference, policy.network_values(patch["cell_ids"])):
            raise ValueError("repair-patch reference differs from the network values")
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    policy = convert(arguments.source)
    policy.save(arguments.destination)
    print(f"{arguments.source} -> {arguments.destination}: grid {policy.sizes.tolist()}, "
          f"{len(policy.overrides)} overrides, repair values {policy.min_value}..{policy.max_value}")


if __name__ == "__main__":
    main()
