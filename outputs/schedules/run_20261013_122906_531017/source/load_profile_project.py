"""Shared residential Flow Matching model, constraints and evaluation.

A sample is a 24-hour UTC household profile. prepare_data returns original
CSV units; training and sampling use y=x/scale without centering. Evaluation
converts back to original units before reporting metrics. The same trained
model is reused across A/B/C/D. Importing this module does not start training.
"""

from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


class Velocity(nn.Module):
    """Ambient-space velocity MLP conditioned on scalar flow time.

    The default is a 24-dimensional profile. Although the network dimension
    is configurable, the residential constraints and metrics assume 24 hours.
    """

    def __init__(self, dimension=24):
        """Build four 256-unit SiLU hidden layers with a time-augmented input."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension + 1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, dimension),
        )

    def forward(self, x, t):
        """Return velocities with the same shape as x: (batch, dimension).

        Concatenate t of shape (batch, 1) as an extra feature. x and t must match
        the network's dtype/device and use normalized consumption coordinates.
        """
        return self.net(torch.cat((x, t), dim=1))


def prepare_data(path="fluvius_wide_format.csv", seed=42):
    """Read the real CSV and return raw daily arrays plus preprocessing metadata.

    path points to a wide CSV with UTC timestamps in its first column and one
    household per remaining column. Validate complete, aligned hourly days;
    reshape into household-days and remove days with negative/nonfinite values.
    Split household identities 70/15/15 using seed, so one household cannot
    appear in multiple splits. Days may still share dates and be dependent.

    Return (arrays, info): arrays maps train/validation/test to float32 arrays
    of shape (n_days, 24), still in ORIGINAL units. info records split IDs,
    counts, SHA-256, training-only standard deviation and the training 99.5th
    percentile of absolute within-day ramps. Normalization is performed later;
    the empirical ramp bound is not asserted to be a physical law.
    """
    frame = pd.read_csv(path, index_col=0)
    index = pd.to_datetime(frame.index, utc=True)
    if len(frame) == 0 or not (index == index.floor("h")).all():
        raise ValueError("Expected nonempty timestamps aligned to full UTC hours")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Timestamps must be ordered and unique")
    if not np.all(index[1:] - index[:-1] == pd.Timedelta(hours=1)):
        raise ValueError(
            "Missing or non-hourly timestamps: inspect before forming days"
        )
    if index[0].hour != 0 or index[-1].hour != 23 or len(frame) % 24:
        raise ValueError("Expected complete UTC days")
    values = frame.to_numpy(dtype=np.float32)
    # Arrange data as (household, day, hour) before splitting identities.
    # Splitting flattened days first could place the same household in both sets.
    daily = values.reshape(-1, 24, values.shape[1]).transpose(2, 0, 1)
    ids = np.random.default_rng(seed).permutation(len(frame.columns))
    n_train, n_val = round(0.70 * len(ids)), round(0.15 * len(ids))
    groups = dict(
        train=ids[:n_train],
        validation=ids[n_train : n_train + n_val],
        test=ids[n_train + n_val :],
    )
    arrays, counts = {}, {}
    for name, indices in groups.items():
        a = daily[indices].reshape(-1, 24)
        valid = np.isfinite(a).all(axis=1) & (a >= 0).all(axis=1)
        arrays[name] = a[valid]
        counts[name] = dict(
            households=len(indices),
            retained_days=int(valid.sum()),
            removed_days=int((~valid).sum()),
        )
    # A single positive scale preserves geometry and the nonnegative boundary.
    scale = float(arrays["train"].std())
    if not scale > 0:
        raise ValueError("Degenerate training data")
    # Estimate the operational ramp limit on training days only; test peaks
    # remain in the reference distribution even when they violate this limit.
    ramp = float(np.quantile(np.abs(np.diff(arrays["train"], axis=1)), 0.995))
    if not ramp > 0:
        raise ValueError("Degenerate ramp threshold")
    info = dict(
        source="https://zenodo.org/records/20525554",
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        shape=list(values.shape),
        time_basis="UTC, complete 24-hour days",
        sample="one household-day, 24 hourly consumption values",
        units="original CSV units; physical unit not specified on Zenodo record",
        split_seed=seed,
        split_households={k: frame.columns[v].tolist() for k, v in groups.items()},
        counts=counts,
        scale=scale,
        ramp_threshold=ramp,
        ramp_quantile=0.995,
        negative_values=int((values < 0).sum()),
        nonfinite_values=int((~np.isfinite(values)).sum()),
    )
    return arrays, info


def train_model(train, validation, scale, out, steps=5000, seed=42):
    """Fit one unconstrained model and select its checkpoint using validation MSE.

    train and validation are (n_days, 24) float32 arrays in original units.
    scale is the positive training-only standard deviation. out must be an
    existing directory. steps counts optimizer iterations, not full epochs;
    seed controls initialization and training randomness. Fixed validation
    noise/times use a separate generator, never the test distribution.

    Return (model, history, selected_step). Save model.pt and training.csv in
    out. The returned model contains the best validation weights in evaluation
    mode. The checkpoint supports inference, not exact optimizer/RNG resumption.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Velocity().to(device)
    data = torch.as_tensor(train / scale, device=device)
    # Fixed validation regression examples, never used for gradients.
    rng = np.random.default_rng(31415)
    val = torch.as_tensor(
        validation[
            rng.choice(len(validation), min(4096, len(validation)), replace=False)
        ]
        / scale,
        device=device,
    )
    g = torch.Generator(device=device).manual_seed(31415)
    noise = torch.randn(val.shape, generator=g, device=device)
    vt = torch.rand((len(val), 1), generator=g, device=device)
    vx, target = (1 - vt) * noise + vt * val, val - noise
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best, best_state, best_step = [], float("inf"), None, None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        # Re-sample source noise and interpolation time for each batch.
        # The target velocity is constant along each conditional straight path.
        x1 = data[torch.randint(len(data), (512,), device=device)]
        x0, t = torch.randn_like(x1), torch.rand(512, 1, device=device)
        loss = (model((1 - t) * x0 + t * x1, t) - (x1 - x0)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vloss = (model(vx, vt) - target).square().mean().item()
            history.append(
                dict(step=step, train_loss=loss.item(), validation_loss=vloss)
            )
            if vloss < best:
                best, best_step = vloss, step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"Step {step}/{steps}: train={loss.item():.4f}, validation={vloss:.4f}, elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        dict(
            model_state_dict=best_state,
            scale=scale,
            seed=seed,
            selected_step=best_step,
            steps=steps,
            architecture="25 -> 256 x 4 SiLU -> 24",
        ),
        out / "model.pt",
    )
    history = pd.DataFrame(history)
    history.to_csv(out / "training.csv", index=False)
    return model, history, best_step


def constraint_matrix(ramp, device, dtype, constraints="both"):
    """Build G and b for feasible normalized rows satisfying y @ G.T <= b.

    ramp is expressed in the SAME units as y. 'both' returns 70 inequalities:
    24 nonnegative bounds and 46 signed adjacent-hour differences. 'nonnegative'
    and 'ramp' select subsets for the ablation. There is no last-to-first-hour
    constraint. Allocate the matrix using the requested torch device and dtype.
    """
    eye = torch.eye(24, device=device, dtype=dtype)
    diff = eye[1:] - eye[:-1]
    # Feasible iff G x - b <= 0 (in normalized units).
    G = torch.cat((-eye, diff, -diff))
    b = torch.cat(
        (
            torch.zeros(24, device=device, dtype=dtype),
            torch.full((46,), ramp, device=device, dtype=dtype),
        )
    )
    if constraints == "nonnegative":
        return G[:24], b[:24]
    if constraints == "ramp":
        return G[24:], b[24:]
    if constraints != "both":
        raise ValueError("constraints must be both, nonnegative or ramp")
    return G, b


def soft_correction(x, ramp, strength=0.3, constraints="both"):
    # Gradient descent on 1/2 ||relu(Gx-b)||^2 / ||G||_2^2.
    # ||G||_2^2 <= 9 for nonnegativity and both signed adjacent differences.
    """Take one gradient step on a squared hinge penalty for active inequalities.

    x has shape (batch, 24); ramp uses the same normalized units. The gradient
    of 0.5*||relu(Gx-b)||^2 is G.T@relu(Gx-b). The fixed divisor 9 bounds ||G||^2
    for the full constraint matrix and is retained in subset ablations to avoid
    confounding constraint removal with a different step scaling. This soft
    step reduces violations under the chosen strength; it need not reach zero.
    """
    G, b = constraint_matrix(ramp, x.device, x.dtype, constraints)
    return x - (strength / 9.0) * (torch.relu(x @ G.T - b) @ G)


def project_profiles(x, ramp, tolerance=2e-6, max_cycles=2000, constraints="both"):
    """Compute a Euclidean projection of (batch, 24) profiles onto active bounds.

    x and ramp must use the same units. For 'both', Dykstra alternates three
    convex sets: nonnegativity, even adjacent-pair ramp slabs, and odd slabs.
    Pairs within each slab set are disjoint, giving exact subprojections.
    Dykstra's correction vectors are essential: simple repeated clipping would
    not generally yield the nearest point in the intersection. 'ramp' omits
    the orthant; 'nonnegative' is exactly clamp_min(0).

    Return a new tensor, preserving dtype/device. Stop when both iterate change
    and maximum violation are at most tolerance, or raise on nonconvergence.
    This numerical stopping rule is checked against an independent optimizer;
    feasibility alone is not an exact optimality certificate.
    """
    if constraints not in ("both", "nonnegative", "ramp"):
        raise ValueError("Unknown constraints")
    if ramp <= 0 or tolerance <= 0 or max_cycles <= 0:
        raise ValueError("ramp, tolerance and max_cycles must be positive")
    if constraints == "nonnegative":
        return x.clamp_min(0)
    z = x.clone()
    corrections = [torch.zeros_like(z) for _ in range(3)]
    for cycle in range(max_cycles):
        old = z.clone()
        for k in ([0, 1, 2] if constraints == "both" else [1, 2]):
            y = z + corrections[k]
            if k == 0:
                z = y.clamp_min(0)
            else:
                z = y.clone()
                start = k - 1
                left = y[:, start:23:2]
                right = y[:, start + 1 : 24 : 2]
                delta = right - left
                adjustment = (delta - delta.clamp(-ramp, ramp)) / 2
                z[:, start:23:2] = left + adjustment
                z[:, start + 1 : 24 : 2] = right - adjustment
            # Carry the residual into the next cycle: this distinguishes
            # Dykstra from alternating projections onto intersecting sets.
            corrections[k] = y - z
        if (z - old).abs().max().item() <= tolerance:
            violation = max(
                torch.relu(-z).max().item() if constraints == "both" else 0.0,
                torch.relu(torch.diff(z, dim=1).abs() - ramp).max().item(),
            )
            if violation <= tolerance:
                return z
    raise RuntimeError("Dykstra did not converge; increase max_cycles or inspect input")


@torch.no_grad()
def sample_four(model, initial, ramp, steps=100, constraints="both", trace=None):
    """Sample all four methods from a shared normalized initial tensor.

    initial has shape (batch, 24), and ramp is the ORIGINAL ramp divided by
    scale. steps must be a positive integer. constraints selects the active
    set, defaulting to the main joint protocol. Return tensors A/B/C/D in
    normalized units, without changing initial. The model is set to eval mode.

    Optional trace(time, method, state, correction, event) observes each
    post-step state and D's separate terminal projection; it must not mutate
    its tensors. Corrections are post-correction minus pre-correction states.
    Initial-state records, if desired, are added by the caller. D uses A's
    trajectory until its final projection, avoiding a redundant network pass.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    model.eval()
    a, b, c = (initial.clone() for _ in range(3))
    for k in range(steps):
        state = torch.cat((a, b, c))
        t = torch.full(
            (len(state), 1), k / steps, device=state.device, dtype=state.dtype
        )
        a, b, c = (state + model(state, t) / steps).chunk(3)
        raw_b, raw_c = b, c
        b = soft_correction(b, ramp, constraints=constraints)
        c = project_profiles(c, ramp, constraints=constraints)
        if trace is not None:
            for method, state, correction in [
                ("A", a, torch.zeros_like(a)),
                ("B", b, b - raw_b),
                ("C", c, c - raw_c),
                ("D", a, torch.zeros_like(a)),
            ]:
                trace((k + 1) / steps, method, state, correction, "post_step")
    result = dict(A=a, B=b, C=c, D=project_profiles(a, ramp, constraints=constraints))
    if trace is not None:
        trace(1.0, "D", result["D"], result["D"] - a, "terminal_projection")
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise RuntimeError("Non-finite samples")
    return result


def autocorrelation(x, max_lag=6):
    """Return mean within-day autocorrelation for lags 1 through max_lag.

    x has shape (n_profiles, 24). Centre each profile by its own daily mean;
    use its full sum of squared deviations as denominator for every lag.
    Exclude nearly constant profiles and return zeros if none remain.
    The result is dimensionless, not cross-day or population autocorrelation.
    """
    centered = x - x.mean(axis=1, keepdims=True)
    denom = np.sum(centered**2, axis=1)
    nonconstant = denom > 1e-12
    if not nonconstant.any():
        return np.zeros(max_lag)
    c, d = centered[nonconstant], denom[nonconstant]
    return np.array(
        [
            (np.sum(c[:, :-lag] * c[:, lag:], axis=1) / d).mean()
            for lag in range(1, max_lag + 1)
        ]
    )


def metrics(x, reference, ramp, directions, tolerance):
    """Compare equally sized (n, 24) profiles and references in original units.

    ramp and tolerance use those same units; directions has shape (24, k)
    with unit columns. Feasibility always evaluates BOTH constraints, including
    when the sampler's active ablation set is smaller. Return fractions,
    violation magnitudes, SW1, mean-shape/variability errors, peak/total W1 and
    ACF error. Violation counts combine 24 lower bounds and 23 absolute ramps;
    they do not count the two signed versions of each ramp separately.
    """
    lower = np.maximum(-x, 0)
    excess = np.maximum(np.abs(np.diff(x, axis=1)) - ramp, 0)
    both = np.concatenate((lower, excess), axis=1)
    return dict(
        feasible_fraction=float((both.max(axis=1) <= tolerance).mean()),
        negative_hour_fraction=float((lower > tolerance).mean()),
        ramp_violation_fraction=float((excess > tolerance).mean()),
        mean_violations_per_profile=float((both > tolerance).sum(axis=1).mean()),
        mean_lower_violation=float(lower.mean()),
        mean_ramp_excess=float(excess.mean()),
        sliced_w1=float(
            np.abs(
                np.sort(x @ directions, axis=0)
                - np.sort(reference @ directions, axis=0)
            ).mean()
        ),
        mean_shape_rmse=float(np.sqrt(np.mean((x.mean(0) - reference.mean(0)) ** 2))),
        hourly_std_rmse=float(np.sqrt(np.mean((x.std(0) - reference.std(0)) ** 2))),
        peak_w1=float(wasserstein_distance(x.max(1), reference.max(1))),
        daily_total_w1=float(wasserstein_distance(x.sum(1), reference.sum(1))),
        acf_mae=float(np.abs(autocorrelation(x) - autocorrelation(reference)).mean()),
    )


def evaluate(model, arrays, info, out, n=2000, seeds=range(10), steps=100):
    """Evaluate one fitted model over paired sampling seeds against test data.

    arrays/info come from prepare_data; out is an existing output directory.
    For each seed, A/B/C/D share noise and reference points. Two disjoint test
    subsets of size n provide generated/reference and reference/reference
    comparisons. Convert outputs to original units before computing metrics.

    Write per-seed NPZ samples, metrics.csv, summary.csv and reference_metrics.csv.
    Return (frame, summary, illustration), where illustration contains reference
    and method samples for seed 0; callers requesting plots must include seed 0.
    SD across these seeds measures sampling/reference variation, not training
    uncertainty. At least 2*n test profiles are required.
    """
    seeds = tuple(seeds)
    if n <= 0 or 2 * n > len(arrays["test"]) or not seeds:
        raise ValueError(
            "Need positive n, nonempty seeds and at least 2*n test samples"
        )
    scale, ramp = info["scale"], info["ramp_threshold"]
    device = next(model.parameters()).device
    directions = np.random.default_rng(2026).normal(size=(24, 128))
    directions /= np.linalg.norm(directions, axis=0)
    tolerance = max(1e-5, scale * 1e-5)
    rows, baseline = [], []
    for seed in seeds:
        start = time.perf_counter()
        generator = torch.Generator(device=device).manual_seed(seed)
        initial = torch.randn(n, 24, generator=generator, device=device)
        rng = np.random.default_rng(10000 + seed)
        indices = rng.choice(len(arrays["test"]), 2 * n, replace=False)
        ref, ref2 = arrays["test"][indices[:n]], arrays["test"][indices[n:]]
        samples = {
            k: v.cpu().numpy() * scale
            for k, v in sample_four(model, initial, ramp / scale, steps).items()
        }
        for name, x in samples.items():
            rows.append(
                dict(
                    seed=seed,
                    method=name,
                    **metrics(x, ref, ramp, directions, tolerance),
                )
            )
        baseline.append(
            dict(seed=seed, **metrics(ref2, ref, ramp, directions, tolerance))
        )
        np.savez_compressed(out / f"samples_seed_{seed}.npz", reference=ref, **samples)
        if seed == 0:
            illustration = (ref, samples)
        print(f"Application seed {seed}: {time.perf_counter()-start:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False)
    summary = frame.groupby("method")[
        [c for c in frame if c not in ("seed", "method")]
    ].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    pd.DataFrame(baseline).to_csv(out / "reference_metrics.csv", index=False)
    return frame, summary, illustration


def plot_results(history, illustration, frame, out):
    """Save and display primary diagnostics for an evaluated residential run.

    history contains training/validation losses; illustration is the seed-0
    (reference, samples) tuple returned by evaluate; frame contains all seeds.
    Example profiles/CDFs illustrate one seed, while trade-off bars aggregate
    all seeds with sample SD. Files in out are derived plots and may be replaced.
    """
    from notebook_utils import COLORS

    ref, samples = illustration
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.step, history.train_loss, label="Training batch")
    ax.plot(history.step, history.validation_loss, label="Fixed validation")
    ax.set(xlabel="Iteration", ylabel="Flow Matching MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=160)
    plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for name, x in dict(Reference=ref, **samples).items():
        axes[0].plot(range(24), x.mean(0), label=name, color=COLORS[name],
                     linestyle="--" if name == "Reference" else "-")
        axes[1].plot(range(24), x.std(0), label=name, color=COLORS[name],
                     linestyle="--" if name == "Reference" else "-")
        axes[2].plot(range(1, 7), autocorrelation(x), label=name, color=COLORS[name],
                     linestyle="--" if name == "Reference" else "-")
    for ax, title in zip(
        axes,
        ["Mean hourly profile", "Hourly variability", "Within-day autocorrelation"],
    ):
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    axes[0].set(xlabel="UTC hour", ylabel="Consumption (CSV units)")
    axes[1].set(xlabel="UTC hour", ylabel="Standard deviation (CSV units)")
    axes[2].set(xlabel="Lag (hours)", ylabel="Autocorrelation")
    fig.tight_layout()
    fig.savefig(out / "fidelity.png", dpi=160)
    plt.show()
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.3), sharey=True)
    for ax, (name, x) in zip(axes, dict(Reference=ref, **samples).items()):
        ax.plot(np.arange(24), x[:12].T, alpha=0.45, color=COLORS[name])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set(title=name, xlabel="UTC hour")
    axes[0].set_ylabel("Consumption (CSV units)")
    fig.tight_layout()
    fig.savefig(out / "profiles.png", dpi=160)
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, x in dict(Reference=ref, **samples).items():
        for ax, values in zip(axes, [x.sum(1), x.max(1)]):
            ordered = np.sort(values)
            ax.plot(ordered, np.arange(1, len(x) + 1) / len(x), label=name, color=COLORS[name],
                     linestyle="--" if name == "Reference" else "-")
    for ax, title in zip(axes, ["Daily total", "Daily peak"]):
        ax.set(title=title, xlabel="CSV units", ylabel="Empirical CDF")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "totals_peaks.png", dpi=160)
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ["feasible_fraction", "sliced_w1"]):
        means = frame.groupby("method")[key].mean()
        std = frame.groupby("method")[key].std()
        ax.bar(means.index, means, yerr=std, capsize=4, color=[COLORS[m] for m in means.index])
        ax.set(title="Feasible fraction ↑" if key == "feasible_fraction" else "SW1 ↓",
               xlabel="Method", ylabel="Mean ± sample SD across 10 seeds")
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=160)
    plt.show()
