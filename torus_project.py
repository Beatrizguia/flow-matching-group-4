"""Flow Matching on a surface-uniform torus: shared implementation.

The notebook and numerical tests import this module so that the model,
constraint corrections and metrics have one implementation. Importing the
module does not train a model or generate results. Sampling uses torch tensors;
reference generation and distribution metrics use NumPy arrays. Unless stated
otherwise, the geometry is fixed at major radius R=3 and tube radius r=1.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


class FlowMatchingNetwork(nn.Module):
    """Velocity field v_theta(x, t) for three-dimensional ambient-space transport.

    The prescribed MLP has four 256-unit SiLU hidden layers. Geometry is enforced
    by the sampler, not by the network architecture or training objective.
    """

    def __init__(self):
        """Construct the prescribed 4-input, 3-output velocity network."""
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x, t):
        # x has shape (batch_size, 3)
        # t has shape (batch_size, 1)

        """Predict a velocity tensor with shape (batch, 3).

        x contains ambient coordinates with shape (batch, 3); t contains one flow
        time per point with shape (batch, 1). Both must match the model dtype/device.
        """
        xt = torch.cat([x, t], dim=1)

        return self.network(xt)


def sample_torus(n_samples, R=3.0, r=1.0):
    """Generate n_samples points uniformly with respect to torus surface area.

    Returns a NumPy array of shape (n_samples, 3). For R > r > 0, the surface
    Jacobian is proportional to R + r*cos(phi); rejection sampling therefore
    corrects the otherwise nonuniform surface density of uniform angles.
    This training-data helper uses NumPy's global RNG, seeded by the notebook.
    """
    theta = np.random.uniform(0, 2 * np.pi, n_samples)

    phi = []

    while len(phi) < n_samples:
        candidate = np.random.uniform(0, 2 * np.pi)

        acceptance_probability = (R + r * np.cos(candidate)) / (R + r)

        if np.random.rand() < acceptance_probability:
            phi.append(candidate)

    phi = np.array(phi)

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    samples = np.stack([x, y, z], axis=1)

    return samples


R, r = 3.0, 1.0


def torus_h(x):
    """Return the signed equality residual h(x), shape (batch,).

    Input x has shape (batch, 3). h=0 describes the torus; h is a quadratic
    constraint residual, not the Euclidean distance to the surface.
    """
    rho = torch.linalg.vector_norm(x[:, :2], dim=1)
    return (rho - R).square() + x[:, 2].square() - r**2


def radial_direction(x):
    """Return xy radius (batch, 1) and radial unit direction (batch, 2).

    On the z-axis the radial direction is undefined. Select +x deterministically
    and avoid division by zero; this is a numerical convention at a singularity.
    """
    rho = torch.linalg.vector_norm(x[:, :2], dim=1, keepdim=True)
    safe = torch.where(rho > 0, rho, torch.ones_like(rho))
    fallback = torch.zeros_like(x[:, :2])
    fallback[:, 0] = 1
    return rho, torch.where(rho > 0, x[:, :2] / safe, fallback)


def hard_project(x):
    """Project each (batch, 3) point onto the fixed torus surface.

    In the radial-z plane, project onto the tube circle, then restore the xy
    direction. At a tube centre the nearest point is nonunique: select the
    outer side. Preserve tensor dtype/device and leave the input unchanged.
    """
    rho, radial = radial_direction(x)
    tube = torch.cat((rho - R, x[:, 2:3]), dim=1)
    length = torch.linalg.vector_norm(tube, dim=1, keepdim=True)
    fallback = torch.zeros_like(tube)
    fallback[:, 0] = 1
    direction = torch.where(
        length > 0,
        tube / torch.where(length > 0, length, torch.ones_like(length)),
        fallback,
    )
    return torch.cat(
        ((R + r * direction[:, :1]) * radial, r * direction[:, 1:2]), dim=1
    )


def soft_correct(x, strength=0.3, eps=1e-8):
    """Take the prescribed normalized equality correction at updated points.

    Compute x - strength*h(x)*grad(h)/(||grad(h)||^2 + eps), with default
    strength 0.3. eps regularizes the denominator; it does not guarantee exact
    feasibility. The radial gradient convention at the z-axis is inherited
    from radial_direction. Input and output have shape (batch, 3).
    """
    rho, radial = radial_direction(x)
    grad = torch.cat((2 * (rho - R) * radial, 2 * x[:, 2:3]), dim=1)
    return x - strength * torus_h(x)[:, None] * grad / (
        grad.square().sum(dim=1, keepdim=True) + eps
    )


@torch.no_grad()
def sample_abcd(model, x0, n_steps=100):
    """Generate A/B/C/D from one model and paired initial noise.

    x0 is a (batch, 3) tensor compatible with the model. n_steps Euler updates
    span flow time [0, 1]. A follows the learned field; B and C respectively
    apply soft correction and hard projection after every update. D projects
    A's final state without an extra neural trajectory. Return a dictionary
    of tensors. Evaluation mode is enabled, gradients are disabled, and x0
    is not modified. Nonpositive step counts or nonfinite outputs fail.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    model.eval()
    a, b, c = (x0.clone() for _ in range(3))
    for k in range(n_steps):
        # One batched evaluation for the three independent trajectories.
        state = torch.cat((a, b, c), dim=0)
        t = torch.full(
            (len(state), 1), k / n_steps, device=state.device, dtype=state.dtype
        )
        next_state = state + model(state, t) / n_steps
        a, b, c = next_state.chunk(3)
        b = soft_correct(b)
        c = hard_project(c)
    result = {"A": a, "B": b, "C": c, "D": hard_project(a)}
    if not all(torch.isfinite(x).all().item() for x in result.values()):
        raise RuntimeError("Sampling produced NaN or infinity.")
    return result


def reference_torus(n, rng):
    """Draw n surface-uniform reference points with a supplied NumPy Generator.

    Return shape (n, 3). An explicit generator separates evaluation-reference
    randomness from training-data generation and makes seed pairing auditable.
    The vectorized rejection sampler uses the same density as sample_torus.
    """
    theta = rng.uniform(0, 2 * np.pi, n)
    accepted = []
    count = 0
    while count < n:
        candidates = rng.uniform(0, 2 * np.pi, n)
        keep = candidates[rng.random(n) < (R + r * np.cos(candidates)) / (R + r)]
        accepted.append(keep)
        count += len(keep)
    phi = np.concatenate(accepted)[:n]
    return np.column_stack(
        (
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi),
        )
    )


def sw1(x, reference, directions):
    """Estimate sliced Wasserstein-1 for equally sized point clouds.

    x and reference have shape (n, 3); directions has shape (3, n_directions)
    and must contain unit-length columns. Sorting each one-dimensional
    projection gives the empirical W1 for equal sample weights. Average over
    samples and directions; this is a finite-direction estimate, not full W1.
    """
    if x.shape != reference.shape:
        raise ValueError("SW1 here requires equally sized point clouds")
    return float(
        np.abs(
            np.sort(x @ directions, axis=0) - np.sort(reference @ directions, axis=0)
        ).mean()
    )


def measure(x, reference, directions):
    """Return geometric feasibility and distributional fidelity for one cloud.

    Inputs follow sw1. Report mean absolute h, true mean surface distance,
    fraction satisfying |h| <= 1e-3, and sliced W1. Lower errors and higher
    feasible_fraction are preferable. The tolerance is part of this protocol.
    """
    rho = np.linalg.norm(x[:, :2], axis=1)
    tube_distance = np.sqrt((rho - R) ** 2 + x[:, 2] ** 2)
    residual = np.abs(tube_distance**2 - r**2)
    return {
        "mean_abs_h": float(residual.mean()),
        "mean_surface_distance": float(np.abs(tube_distance - r).mean()),
        "feasible_fraction": float((residual <= 1e-3).mean()),
        "sliced_w1": sw1(x, reference, directions),
    }


def torus_angles(x):
    """Recover theta and phi in [0, 2*pi) for an (n, 3) NumPy point cloud.

    The same angular coordinates describe off-surface points relative to the
    ring centre. Degenerate axes inherit NumPy's arctan2 convention; angular
    marginals alone cannot establish agreement of the full 3D distribution.
    """
    return (
        np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi),
        np.mod(np.arctan2(x[:, 2], np.linalg.norm(x[:, :2], axis=1) - R), 2 * np.pi),
    )


def phi_cdf(phi):
    """Evaluate the target surface-area CDF for angles in [0, 2*pi].

    F(phi) = (phi + (r/R)*sin(phi))/(2*pi). The input is assumed to lie within
    this interval; the function does not clamp or periodically extend a CDF.
    """
    phi = np.asarray(phi)
    return (phi + (r / R) * np.sin(phi)) / (2 * np.pi)


def phi_ks(points):
    """Return the two-sided empirical KS discrepancy from the analytic phi CDF.

    Use the fixed angular cut at zero and both sides of each empirical-CDF
    jump. This descriptive statistic is not rotation invariant; no p-value
    is returned. Empty samples raise ValueError.
    """
    phi = np.sort(torus_angles(points)[1])
    if not len(phi):
        raise ValueError("An empty sample has no empirical CDF")
    target = phi_cdf(phi)
    n = len(phi)
    return float(
        max(np.max(np.arange(1, n + 1) / n - target), np.max(target - np.arange(n) / n))
    )


def supplementary_analysis(run_dir):
    """Recompute angular and paired SW1 tables from a saved torus run.

    Read samples_seed_*.npz and metrics.csv in run_dir. Write angular_metrics,
    angular_summary, paired_sw1 and paired_sw1_summary CSVs, and return those
    four DataFrames in that order. Paired differences subtract A within each
    seed. No model is trained; saved samples and checkpoints are unchanged.
    """
    run_dir = Path(run_dir)
    angular = []
    for path in sorted(run_dir.glob("samples_seed_*.npz")):
        seed = int(path.stem.rsplit("_", 1)[1])
        with np.load(path) as saved:
            for method in ["A", "B", "C", "D", "reference"]:
                angular.append(
                    dict(seed=seed, method=method, phi_ks=phi_ks(saved[method]))
                )
    angular = pd.DataFrame(angular)
    if angular.empty:
        raise ValueError("No saved samples found")
    angular.to_csv(run_dir / "angular_metrics.csv", index=False)
    angular_summary = angular.groupby("method")["phi_ks"].agg(["mean", "std"])
    angular_summary.to_csv(run_dir / "angular_summary.csv")
    frame = pd.read_csv(run_dir / "metrics.csv")
    pivot = frame.pivot(index="seed", columns="method", values="sliced_w1")
    if pivot[["A", "B", "C", "D"]].isna().any().any():
        raise ValueError("Incomplete paired methods")
    paired = pd.DataFrame({name + "-A": pivot[name] - pivot["A"] for name in "BCD"})
    paired.index.name = "seed"
    paired.to_csv(run_dir / "paired_sw1.csv")
    paired_summary = paired.agg(["mean", "std"]).T
    paired_summary.index.name = "comparison"
    paired_summary.to_csv(run_dir / "paired_sw1_summary.csv")
    return angular, angular_summary, paired, paired_summary
