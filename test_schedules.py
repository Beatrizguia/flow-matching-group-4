"""Check schedule timing independently, then audit saved paired experiments.

Run without -O. The analytic fixture has a time-dependent velocity: early
clipping changes the endpoint, so an off-by-one error cannot pass by merely
counting calls. Existing A/C/D implementations provide a separate regression
check. Completed outputs, when present, are checked without retraining.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from load_profile_project import Velocity, metrics, sample_four
from schedule_experiments import (
    projection_schedules,
    sample_schedules,
    paired_differences,
)

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)


class TimeField(torch.nn.Module):
    """A known field: Euler increments at four steps are -1, 0, 1, 2."""

    def forward(self, x, t):
        return torch.ones_like(x) * (16 * t - 4)


initial = torch.full((2, 24), -1.0, dtype=torch.float64)
original = initial.clone()
calls = []


def exact_nonnegative(x, ramp):
    """Independent exact projector; constant profiles already satisfy ramps."""
    calls.append(x.clone())
    return x.clamp_min(0)


with patch("schedule_experiments.project_profiles", side_effect=exact_nonnegative):
    samples = sample_schedules(
        TimeField(),
        initial,
        1.0,
        {"early": (1, 4), "late": (3, 4), "free": ()},
        steps=4,
    )
assert len(calls) == 4
assert torch.equal(samples["early"], torch.full_like(initial, 3))
assert torch.equal(samples["late"], torch.full_like(initial, 2))
assert torch.equal(samples["free"], torch.full_like(initial, 1))
assert torch.equal(initial, original)

schedules = projection_schedules()
assert len(schedules["Early25"]) == len(schedules["Late25"]) == 25
assert set(schedules["Early25"]) & set(schedules["Late25"]) == {100}
assert schedules["Early25"] == (*range(1, 25), 100)
assert schedules["Late25"] == tuple(range(76, 101))

# Cross-check against the established sampler with a real randomly initialized
# nonlinear field, using double precision to minimise batching round-off.
torch.manual_seed(73)
model = Velocity().double().eval()
x = torch.randn(7, 24, dtype=torch.float64)
expected = sample_four(model, x, 1.0, steps=4)
observed = sample_schedules(
    model, x, 1.0, {"A": (), "C": (1, 2, 3, 4), "D": (4,)}, steps=4
)
for name in ["A", "C", "D"]:
    torch.testing.assert_close(observed[name], expected[name], rtol=1e-10, atol=1e-10)
trace = []
with_trace = sample_schedules(
    model,
    x,
    1.0,
    {"A": (), "C": (1, 2, 3, 4), "D": (4,)},
    steps=4,
    trace=lambda *args: trace.append(args),
)
for name in observed:
    assert torch.equal(with_trace[name], observed[name])
assert len(trace) == 12
for invalid in [(0,), (5,), (1, 1), (True,), (1.5,)]:
    try:
        sample_schedules(model, x, 1.0, {"bad": invalid}, steps=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid schedule accepted")

manifest = ROOT / "schedule_run.json"
if manifest.exists():
    out = ROOT / json.loads(manifest.read_text())["run_dir"]
    config = json.loads((out / "protocol.json").read_text())
    assert json.loads((out / "COMPLETE.json").read_text())["rows"] == 150
    frame = pd.read_csv(out / "metrics.csv")
    assert not frame.duplicated(["training_seed", "seed", "method"]).any()
    assert len(frame) == 150
    assert set(frame.training_seed) == {42, 43, 44} and set(frame.seed) == set(
        range(10)
    )
    assert frame.groupby(["training_seed", "seed"]).method.nunique().eq(5).all()
    assert frame.groupby("method").projection_calls.first().to_dict() == {
        k: len(v) for k, v in schedules.items()
    }
    assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
    for name, digest in config["code_sha256"].items():
        assert (
            hashlib.sha256((out / "source" / name).read_bytes()).hexdigest() == digest
        )
    for cp in config["checkpoints"].values():
        assert (
            hashlib.sha256((ROOT / cp["path"]).read_bytes()).hexdigest() == cp["sha256"]
        )
    info = json.loads((out / "data_config.json").read_text())
    directions = np.random.default_rng(2026).normal(size=(24, 128))
    directions /= np.linalg.norm(directions, axis=0)
    # Recompute every metric from the 30 saved sample sets, not only summaries.
    for (training_seed, seed), group in frame.groupby(["training_seed", "seed"]):
        with np.load(
            out / f"samples_training_{training_seed}_seed_{seed}.npz"
        ) as saved:
            for row in group.itertuples():
                actual = metrics(
                    saved[row.method],
                    saved["reference"],
                    info["ramp_threshold"],
                    directions,
                    config["feasibility_tolerance_csv"],
                )
                for key, value in actual.items():
                    np.testing.assert_allclose(
                        value, getattr(row, key), atol=1e-10, rtol=1e-10
                    )
    expected = paired_differences(frame).reset_index()
    pd.testing.assert_frame_equal(
        pd.read_csv(out / "paired_late_minus_early.csv"),
        expected,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    expected = (
        frame.groupby(["training_seed", "method"])[
            ["feasible_fraction", "sliced_w1", "daily_total_w1", "mean_consumption"]
        ]
        .mean()
        .reset_index()
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(out / "per_training_seed.csv"),
        expected,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    print(
        "PASS: complete 150-row experiment, source/checkpoint hashes, all saved-sample metrics, paired/model summaries."
    )
else:
    print(
        "Saved-run audit skipped: execute schedule_experiments.py to create schedule_run.json."
    )
print(
    "PASS: analytic timing/count, schedule validation, A/C/D equivalence, immutable input and nonintrusive trace."
)
