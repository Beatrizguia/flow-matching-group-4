"""Executable checks for residential geometry, data splitting and metrics.

Run from the project directory without Python's -O flag. Projection is
compared with an independent SLSQP optimizer, rather than only tested for
feasibility. The second half needs the verified real CSV and checks held-out
households, train-only threshold estimation and zero-error metric identities.
"""

import numpy as np
import torch
from scipy.optimize import minimize, LinearConstraint
from load_profile_project import (
    project_profiles,
    soft_correction,
    constraint_matrix,
    sample_four,
    prepare_data,
    metrics,
)

torch.set_num_threads(2)
torch.manual_seed(8)
x = torch.randn(12, 24, dtype=torch.float64) * 3
ramp = 0.8
z = project_profiles(x, ramp, tolerance=1e-9, max_cycles=10000)
G, b = constraint_matrix(ramp, x.device, x.dtype)
assert (z @ G.T - b).max() < 1e-8
assert torch.allclose(project_profiles(z, ramp, tolerance=1e-9), z, atol=1e-8)

# Check nearest-point optimality against a separate constrained solver.
for i in range(3):
    original = x[i].numpy()
    result = minimize(
        lambda y: 0.5 * np.sum((y - original) ** 2),
        np.zeros(24),
        jac=lambda y: y - original,
        constraints=LinearConstraint(G.numpy(), -np.inf, b.numpy()),
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 2000},
    )
    assert result.success, result.message
    assert np.max(np.abs(result.x - z[i].numpy())) < 2e-6

# Soft correction should decrease its penalty, but need not be fully feasible.
before = torch.relu(x @ G.T - b).square().sum(1)
after = torch.relu(soft_correction(x, ramp) @ G.T - b).square().sum(1)
assert (after <= before + 1e-10).all()
feasible = torch.ones(4, 24, dtype=torch.float64)
assert torch.equal(soft_correction(feasible, ramp), feasible)


class Constant(torch.nn.Module):
    """Deterministic test field whose Euler trajectory is known analytically."""

    def forward(self, x, t):
        """Return a constant velocity so A must end at initial + 0.1."""
        return torch.ones_like(x) * 0.1


initial = torch.randn(8, 24, dtype=torch.float64)
untouched = initial.clone()
samples = sample_four(Constant(), initial, ramp, steps=4)
assert torch.equal(initial, untouched)
assert torch.allclose(samples["A"], initial + 0.1)
assert torch.allclose(samples["D"], project_profiles(samples["A"], ramp))
assert (samples["C"] @ G.T - b).max() < 3e-6
assert (samples["D"] @ G.T - b).max() < 3e-6
print(
    "PASS: Dykstra vs independent constrained optimizer, idempotence, feasibility, soft descent, A trajectory and D=A projection."
)


# Verify the statistical split as well as numerical constraint handling.
arrays, info = prepare_data()
groups = {k: set(v) for k, v in info["split_households"].items()}
assert (
    not groups["train"] & groups["validation"]
    and not groups["train"] & groups["test"]
    and not groups["validation"] & groups["test"]
)
assert sum(len(a) for a in arrays.values()) == 474500
assert (
    info["counts"]["train"]["households"] == 910
    and info["counts"]["test"]["households"] == 195
)
assert np.isclose(
    info["ramp_threshold"], np.quantile(np.abs(np.diff(arrays["train"], axis=1)), 0.995)
)
dirs = np.eye(24)
m = metrics(
    arrays["test"][:100], arrays["test"][:100], info["ramp_threshold"], dirs, 1e-5
)
for key in [
    "sliced_w1",
    "mean_shape_rmse",
    "hourly_std_rmse",
    "peak_w1",
    "daily_total_w1",
    "acf_mae",
]:
    assert m[key] == 0
print(
    "PASS: 474500 daily samples, disjoint households, train-only ramp, identity fidelity metrics."
)
print(info["counts"], "scale", info["scale"], "ramp", info["ramp_threshold"])
