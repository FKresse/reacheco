from pathlib import Path

import pytest
import torch

from reacheco.policy import DWCPolicy

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "policy"
CHECKPOINTS = {
    **{f"dwc_b81_seed{seed}": ROOT / f"checkpoints/acc/dwc_b81_seed{seed}.pt" for seed in range(1, 6)},
    "dwc_b159": ROOT / "checkpoints/acc/dwc_b159.pt",
    "dwc_b159_repaired_expected": GOLDEN / "dwc_b159_repaired.pt",
}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def reference_values(artifact, points):
    """Evaluate the DWC on points directly: clamp, thermometer-encode, run the LUTs."""
    grid, network = artifact["grid"], artifact["network"]
    x = torch.minimum(torch.maximum(points, grid["lower"]), grid["upper"])
    bits = torch.cat([x[:, d, None] > row[None] for d, row in enumerate(grid["thresholds"])], 1)
    for mapping, lut in zip(network["mappings"], network["luts"]):
        address = sum(bits[:, mapping[:, k]].long() << k for k in range(mapping.shape[1]))
        bits = lut[torch.arange(len(lut)), address]
    return bits.reshape(len(points), len(network["beta"]), -1).sum(-1).float()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_matches_prototype(name, device):
    golden = torch.load(GOLDEN / f"{name}.pt", weights_only=False)
    policy = DWCPolicy.load(CHECKPOINTS[name], device=device)
    assert policy.max_value == golden["max_value"]
    values = policy.cell_values(torch.arange(policy.num_cells)).cpu()
    assert torch.equal(values.long(), golden["cell_values"])
    natives = torch.arange(policy.max_value + 1, dtype=torch.float32)[:, None]
    assert torch.equal(policy.decode(natives.to(device)).cpu(), golden["decoded"])


@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_repair_alphabet_spans_both_action_clips(name):
    policy = DWCPolicy.load(CHECKPOINTS[name])
    assert policy.min_value <= 0 and policy.max_value >= policy.output_count
    candidates = policy.repair_candidates(11)
    assert candidates[0].item() == policy.min_value and candidates[-1].item() == policy.max_value
    assert torch.equal(candidates, candidates.unique(dim=0))
    low, high = policy.action_clip
    assert policy.decode(candidates[:1]).item() == low and policy.decode(candidates[-1:]).item() == high
    # The range is minimal: one step inward no longer reaches the clips.
    if policy.min_value < 0:
        assert policy.decode(torch.tensor([[policy.min_value + 1.0]])).item() > low
    if policy.max_value > policy.output_count:
        assert policy.decode(torch.tensor([[policy.max_value - 1.0]])).item() < high


@pytest.mark.parametrize("name", ["dwc_b81_seed1", "dwc_b159"])
def test_cell_convention_matches_pointwise_evaluation(name):
    artifact = torch.load(CHECKPOINTS[name], weights_only=True)
    policy = DWCPolicy(artifact)
    generator = torch.Generator().manual_seed(0)
    span = policy.upper - policy.lower
    random = policy.lower - 0.1 * span + 1.2 * span * torch.rand((4000, 2), generator=generator)
    # Points exactly on thresholds belong to the cell on their left.
    on_threshold = torch.stack([row[torch.randint(len(row), (4000,), generator=generator)]
                                for row in artifact["grid"]["thresholds"]], 1)
    for points in (random, on_threshold, torch.stack((on_threshold[:, 0], random[:, 1]), 1)):
        assert torch.equal(policy.cell_values(policy.cell_ids(points)), reference_values(artifact, points))


def test_box_cell_ids_use_closed_upper_corner():
    policy = DWCPolicy.load(CHECKPOINTS["dwc_b81_seed1"])
    t0, t1 = policy.thresholds[0], policy.thresholds[1]
    boxes = torch.tensor([
        [[t0[2], t1[4]], [t0[3], t1[5]]],                     # interior cell (3, 5)
        [[t0[-2], t1[-2]], [float("inf"), float("inf")]],     # outermost cell
        [[-float("inf"), -float("inf")], [t0[0], t1[0]]],     # innermost cell
    ])
    expected = torch.tensor([3 + 5 * 82, 81 + 81 * 82, 0])
    assert torch.equal(policy.box_cell_ids(boxes), expected)


def test_repaired_artifact_round_trip(tmp_path):
    base = DWCPolicy.load(CHECKPOINTS["dwc_b159"])
    repaired = DWCPolicy.load(CHECKPOINTS["dwc_b159_repaired_expected"])
    assert repaired.changed_cells(base) == frozenset(repaired.overrides)
    assert len(repaired.overrides) == 26

    repaired.set_cell_values([7, 8], [[repaired.min_value], [repaired.max_value]])
    reloaded = DWCPolicy.load(repaired.save(tmp_path / "policy.pt"))
    assert reloaded.overrides == repaired.overrides
    cells = torch.arange(base.num_cells)
    assert torch.equal(reloaded.cell_values(cells), repaired.cell_values(cells))
    assert torch.equal(reloaded.network_values(cells), base.cell_values(cells))


def test_setting_the_network_value_removes_the_override():
    policy = DWCPolicy.load(CHECKPOINTS["dwc_b81_seed1"])
    original = policy.cell_values([10])
    policy.set_cell_values([10], original + 1)
    assert set(policy.overrides) == {10}
    policy.set_cell_values([10], original)
    assert policy.overrides == {}


def test_invalid_values_are_rejected():
    policy = DWCPolicy.load(CHECKPOINTS["dwc_b81_seed1"])
    for values in ([[0.5]], [[policy.min_value - 1.0]], [[policy.max_value + 1.0]], [[float("nan")]]):
        with pytest.raises(ValueError):
            policy.decode(torch.tensor(values))
    with pytest.raises(ValueError):
        policy.set_cell_values([3, 3], [[1], [2]])
    with pytest.raises(IndexError):
        policy.cell_values([policy.num_cells])
