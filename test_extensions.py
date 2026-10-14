"""Regression checks for constraint subsets, observational tracing and guards.

Run without Python's -O flag. Compare all three projections with independent
SLSQP solutions, check ramp-only invariants, and ensure tracing does not
change samples. Synthetic fixtures exercise malformed timestamps/step counts;
no training or real dataset download is needed.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, LinearConstraint
from load_profile_project import (
    project_profiles,
    constraint_matrix,
    sample_four,
    prepare_data,
)

torch.set_num_threads(2)
rng = np.random.default_rng(832)
x = torch.tensor(rng.normal(size=(5, 24)) * 3, dtype=torch.float64)
for mode in ["both", "nonnegative", "ramp"]:
    G, b = constraint_matrix(0.8, "cpu", torch.float64, mode)
    z = project_profiles(x, 0.8, tolerance=1e-10, max_cycles=10000, constraints=mode)
    assert (z @ G.T - b).max() < 1e-8
    for i in range(3):
        original = x[i].numpy()
        result = minimize(
            lambda v: 0.5 * np.sum((v - original) ** 2),
            np.zeros(24),
            jac=lambda v: v - original,
            constraints=LinearConstraint(G.numpy(), -np.inf, b.numpy()),
            method="SLSQP",
            options={"ftol": 1e-11, "maxiter": 2000},
        )
        assert result.success, result.message
        np.testing.assert_allclose(z[i].numpy(), result.x, atol=3e-6)
# A ramp-only projection must not impose positivity, and preserves the mean.
negative = -torch.ones(2, 24, dtype=torch.float64)
assert torch.equal(project_profiles(negative, 0.8, constraints="ramp"), negative)
assert torch.allclose(
    project_profiles(x, 0.8, constraints="ramp").mean(1), x.mean(1), atol=1e-10
)


class Constant(torch.nn.Module):
    """Deterministic test field whose Euler trajectory is known analytically."""

    def forward(self, x, t):
        """Return a constant velocity so A must end at initial + 0.1."""
        return torch.ones_like(x) * 0.1


trace = []
plain = sample_four(Constant(), x, 0.8, steps=4)
observed = sample_four(
    Constant(), x, 0.8, steps=4, trace=lambda *args: trace.append(args)
)
for key in plain:
    assert torch.equal(plain[key], observed[key])
assert (
    len(trace) == 17 and trace[-1][1] == "D" and trace[-1][-1] == "terminal_projection"
)
for steps in [0, -1, 1.5, True]:
    try:
        sample_four(Constant(), x, 0.8, steps=steps)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid step count accepted")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "offset.csv"
    pd.DataFrame(
        rng.random((48, 10)),
        index=pd.date_range("2022-01-01 00:30", periods=48, freq="h", tz="UTC"),
    ).to_csv(path)
    try:
        prepare_data(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Misaligned timestamps accepted")
print(
    "PASS: three projections vs SLSQP, ramp mean preservation, nonintrusive trajectory trace, input validation."
)


# A valid evaluation subset can omit seed 0; only its plot illustration is absent.
from load_profile_project import Velocity, evaluate

with tempfile.TemporaryDirectory() as tmp:
    fixture = np.ones((20, 24), dtype=np.float32)
    rows, summary, illustration = evaluate(
        Velocity().eval(),
        {"test": fixture},
        {"scale": 1.0, "ramp_threshold": 10.0},
        Path(tmp),
        n=4,
        seeds=[3],
        steps=2,
    )
    assert len(rows) == 4 and set(rows.method) == set("ABCD")
    assert illustration is None
print("PASS: evaluation without seed 0 returns measurements and no illustration.")
