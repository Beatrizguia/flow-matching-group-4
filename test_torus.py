"""Executable numerical checks for the prescribed torus experiment.

Run from the project directory with python test_torus.py (without -O, which
disables assertions). Check singular projections, analytic gradients, known
constant-field trajectories, saved sample provenance and angular metrics.
These checks do not retrain the network; supplementary CSVs are regenerated.
"""

import json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(2)
import torus_project

scope = vars(torus_project)

# Include the axis and tube centre, where nearest-point choices are nonunique.
x = torch.tensor(
    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 2.0], [10.0, -5.0, 7.0]],
    dtype=torch.float64,
)
p = scope["hard_project"](x)
assert torch.isfinite(p).all()
assert scope["torus_h"](p).abs().max() < 1e-12
assert torch.allclose(scope["hard_project"](p), p, atol=1e-12)
x = torch.randn(15, 3, dtype=torch.float64, requires_grad=True)
h = scope["torus_h"](x)

# Autograd independently checks the analytic gradient used by strategy B.
gradient = torch.autograd.grad(h.sum(), x)[0]
expected = x - 0.3 * h[:, None] * gradient / (
    gradient.square().sum(1, keepdim=True) + 1e-8
)
assert torch.allclose(scope["soft_correct"](x), expected, atol=1e-12)


class Constant(torch.nn.Module):
    """Deterministic test field whose Euler trajectory is known analytically."""

    def forward(self, x, t):
        """Return a constant velocity so A must end at initial + 0.1."""
        return torch.ones_like(x) * 0.1


noise = torch.randn(5, 3, dtype=torch.float64)
result = scope["sample_abcd"](Constant(), noise, n_steps=10)
assert torch.allclose(result["A"], noise + 0.1)
assert torch.equal(result["D"], scope["hard_project"](result["A"]))
assert scope["torus_h"](result["C"]).abs().max() < 1e-12

# Validate the documented experiment, not merely synthetic unit-test inputs.
latest = Path(json.loads(Path("torus_run.json").read_text())["run_dir"])
config = json.loads((latest / "config.json").read_text())
assert config["training_samples"] == 20000
assert len(list(latest.glob("samples_seed_*.npz"))) == 10
for file in latest.glob("samples_seed_*.npz"):
    data = np.load(file)
    assert all(data[k].shape == (2000, 3) for k in "ABCD")
    assert np.allclose(
        scope["hard_project"](torch.from_numpy(data["A"])).numpy(), data["D"], atol=1e-6
    )
print(
    "PASS: torus degenerate projection, idempotence, B vs autograd, trajectories and all saved 20k-training samples."
)

# Independently check the analytic CDF and KS implementation.
from scipy.stats import kstest
from scipy.integrate import quad
from torus_project import phi_cdf, phi_ks, supplementary_analysis

assert np.isclose(phi_cdf(0), 0) and np.isclose(phi_cdf(2 * np.pi), 1)
for endpoint in [0.5, 2.0, 5.0]:
    integral = quad(lambda v: (3 + np.cos(v)) / (6 * np.pi), 0, endpoint)[0]
    assert np.isclose(phi_cdf(endpoint), integral)
with np.load(latest / "samples_seed_0.npz") as data:
    phi = torus_project.torus_angles(data["A"])[1]
    assert np.isclose(phi_ks(data["A"]), kstest(phi, phi_cdf).statistic)
angular, angular_summary, paired, paired_summary = supplementary_analysis(latest)
assert len(angular) == 50 and paired.shape == (10, 3)
import pandas as pd

metrics = pd.read_csv(latest / "metrics.csv").pivot(
    index="seed", columns="method", values="sliced_w1"
)
for method in "BCD":
    np.testing.assert_allclose(paired[method + "-A"], metrics[method] - metrics["A"])
print(
    "PASS: analytic angular CDF, KS vs SciPy, all-seed angular tables and paired differences."
)
