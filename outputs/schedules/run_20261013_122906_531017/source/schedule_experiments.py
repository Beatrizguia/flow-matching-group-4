"""Compare early and late hard projections at a fixed projection count.

This exploratory validation study reuses the three existing residential models.
It does not retrain, tune a winning schedule, or replace the primary A–D study.
Early25 and Late25 each make 25 calls to the same projection, including the
endpoint. Thus their contrast holds count fixed while changing placement.
Run from any directory: python schedule_experiments.py. A manifest is updated
only after the complete paired experiment and its consistency checks succeed.
"""

from datetime import datetime
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil

import numpy as np
import pandas as pd
import torch

from load_profile_project import Velocity, metrics, prepare_data, project_profiles

ROOT = Path(__file__).resolve().parent
METHODS = ("A", "D", "C", "Early25", "Late25")


def projection_schedules():
    """Return the fixed, one-based post-Euler projection steps (100 steps total).

    Early25: 24 early projections plus the common terminal projection.
    Late25: 24 late intermediate projections plus that terminal projection.
    A/D/C anchor interpretation; their counts differ and are not timing controls.
    """
    return {
        "A": (),
        "D": (100,),
        "C": tuple(range(1, 101)),
        "Early25": (*range(1, 25), 100),
        "Late25": tuple(range(76, 101)),
    }


@torch.no_grad()
def sample_schedules(model, initial, ramp, schedules=None, steps=100, trace=None):
    """Integrate paired normalized profiles, projecting after specified updates.

    initial is a finite floating tensor of shape (batch, 24). Schedule indices
    are unique integers in 1..steps: index k projects after evaluating velocity
    at (k-1)/steps. No projection is made at t=0. Return independent tensors in
    normalized units; never mutate initial. Each schedule gets exactly steps
    velocity evaluations per profile; batching only accelerates those calls.

    The optional observer trace(step, name, state, correction) records post-step
    diagnostics and must not mutate either tensor. All experiments enforce both
    original constraints with the unchanged Dykstra operator and tolerances.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if initial.ndim != 2 or initial.shape[1] != 24 or len(initial) == 0:
        raise ValueError("initial must have nonempty shape (batch, 24)")
    if not initial.is_floating_point() or not torch.isfinite(initial).all():
        raise ValueError("initial must contain finite floating-point values")
    if not np.isfinite(ramp) or ramp <= 0:
        raise ValueError("ramp must be finite and positive")
    schedules = projection_schedules() if schedules is None else schedules
    if not schedules:
        raise ValueError("At least one schedule is required")
    selected = {}
    for name, indices in schedules.items():
        indices = tuple(indices)
        if any(isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= steps
               for k in indices) or len(indices) != len(set(indices)):
            raise ValueError("Projection steps must be unique integers in 1..steps")
        selected[name] = frozenset(indices)
    model.eval()
    states = {name: initial.clone() for name in selected}
    for k in range(1, steps + 1):
        batch = torch.cat(list(states.values()))
        time = torch.full((len(batch), 1), (k - 1) / steps,
                          dtype=batch.dtype, device=batch.device)
        updated = (batch + model(batch, time) / steps).chunk(len(states))
        for (name, indices), raw in zip(selected.items(), updated):
            state = project_profiles(raw, ramp) if k in indices else raw
            states[name] = state
            if trace is not None:
                trace(k, name, state, state - raw)
    if not all(torch.isfinite(x).all() for x in states.values()):
        raise RuntimeError("Non-finite scheduled samples")
    return states


def paired_differences(frame):
    """Subtract Early25 from Late25 within each model and sampling seed.

    Negative SW1/total-W1 differences favour late placement. Pair before taking
    means or SD; the 30 rows are not 30 independent trained models.
    """
    values = ["sliced_w1", "daily_total_w1", "mean_consumption", "feasible_fraction"]
    indexed = frame.set_index(["training_seed", "seed", "method"])
    return (indexed.xs("Late25", level="method")[values]
            - indexed.xs("Early25", level="method")[values])


def plot_schedules(out, frame, trajectories):
    """Save readable schedule and fidelity panels without pooling model SDs.

    SW1 points average ten sampling/reference seeds within each trained model.
    Trajectory panels show the primary seed-42 model, averaging sampling seeds.
    This function only reads measurements; it never trains or samples.
    """
    import matplotlib.pyplot as plt
    from notebook_utils import COLORS

    colours = {**COLORS, "Early25": "#8064A2", "Late25": "#D17A22"}
    schedules = projection_schedules()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, name in enumerate(METHODS):
        indices = schedules[name]
        axes[0].scatter(np.array(indices) / 100, np.full(len(indices), i),
                        marker="|", s=110, color=colours[name])
    axes[0].set(yticks=range(5), yticklabels=[f"{m} ({len(schedules[m])})" for m in METHODS],
                xlabel="Flow time after Euler update", title="Projection schedules (call counts)",
                xlim=(-0.02, 1.04))
    axes[0].invert_yaxis()
    means = frame.groupby(["training_seed", "method"]).sliced_w1.mean()
    for name in METHODS:
        g = means.xs(name, level="method")
        axes[1].plot(g.index, g.values, "o-", label=name, color=colours[name])
    axes[1].set(xlabel="Training seed", xticks=[42, 43, 44], ylabel="Validation SW1 (CSV units)",
                title="Each point averages 10 paired sampling seeds")
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "schedule_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    primary = trajectories[trajectories.training_seed == 42]
    for name in METHODS:
        curve = primary[primary.method == name].groupby("step").mean_consumption.mean()
        ax.plot(curve.index / 100, curve, label=name, color=colours[name])
    ax.set(xlabel="Flow time", ylabel="Mean consumption (CSV units)",
           title="Training seed 42; mean over 10 paired sampling seeds")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(out / "schedule_trajectories.png", dpi=180)
    plt.close(fig)


def main():
    """Run the fixed validation protocol and save samples, metrics and provenance."""
    torch.set_num_threads(4)
    arrays, info = prepare_data(ROOT / "fluvius_wide_format.csv")
    base = ROOT / json.loads((ROOT / "load_run.json").read_text())["run_dir"]
    extra = ROOT / json.loads((ROOT / "supplementary_run.json").read_text())["run_dir"]
    protocol = json.loads((extra / "protocol.json").read_text())
    if (ROOT / protocol["base_run"]).resolve() != base.resolve():
        raise ValueError("Supplementary models belong to a different primary run")
    recorded = json.loads((base / "data_config.json").read_text())
    for key in ["sha256", "scale", "ramp_threshold", "split_households"]:
        if recorded[key] != info[key]:
            raise ValueError(f"Preprocessing mismatch: {key}")
    runs = {42: base, 43: extra / "training_seed_43", 44: extra / "training_seed_44"}
    models = {}
    for seed, run in runs.items():
        # These are trusted local project checkpoints, never arbitrary downloads.
        cp = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        if cp["seed"] != seed or cp["scale"] != info["scale"]:
            raise ValueError("Checkpoint seed or scale mismatch")
        model = Velocity()
        model.load_state_dict(cp["model_state_dict"])
        models[seed] = model.eval()
    out = ROOT / "outputs" / "schedules" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    sources = ["schedule_experiments.py", "load_profile_project.py", "notebook_utils.py"]
    schedules = projection_schedules()
    config = dict(
        base_run=str(base.relative_to(ROOT)), supplementary_run=str(extra.relative_to(ROOT)),
        training_seeds=list(runs), sampling_seeds=list(range(10)), reference_split="validation",
        n_samples=2000, n_steps=100, schedules=schedules, constraints="both",
        projection_tolerance_normalized=2e-6, feasibility_tolerance_csv=max(1e-5, info["scale"]*1e-5),
        sw_seed=2026, sw_directions=128, reference_seed_offset=10000,
        reference_sampling="Two disjoint 2000-profile subsets per seed; shared across all schedules/models",
        primary_contrast="Late25 minus Early25; 25 projection calls each, including endpoint",
        status="Exploratory validation follow-up after historical test inspection; no tuning or new training",
        data_sha256=info["sha256"],
        checkpoints={str(s):dict(path=str((p / "model.pt").relative_to(ROOT)),
                    sha256=hashlib.sha256((p / "model.pt").read_bytes()).hexdigest()) for s,p in runs.items()},
        code_sha256={p:hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        versions={p:importlib.metadata.version(p) for p in ["torch","numpy","pandas","scipy","matplotlib"]},
        python=platform.python_version(),
    )
    # Record choices before observing results. Frozen sources allow later audits.
    (out / "protocol.json").write_text(json.dumps(config, indent=2)+"\n")
    (out / "data_config.json").write_text(json.dumps(info, indent=2)+"\n")
    (out / "source").mkdir()
    for p in sources:
        shutil.copyfile(ROOT / p, out / "source" / p)
    directions = np.random.default_rng(2026).normal(size=(24,128))
    directions /= np.linalg.norm(directions, axis=0)
    scale, ramp = info["scale"], info["ramp_threshold"]
    rows, traces, baseline = [], [], []
    for training_seed, model in models.items():
        for seed in range(10):
            noise = torch.randn(2000,24,generator=torch.Generator().manual_seed(seed))
            indices = np.random.default_rng(10000+seed).choice(len(arrays["validation"]),4000,replace=False)
            reference, reference2 = arrays["validation"][indices[:2000]], arrays["validation"][indices[2000:]]
            if training_seed == 42:
                baseline.append(dict(seed=seed,**metrics(reference2,reference,ramp,directions,config["feasibility_tolerance_csv"])))
            def observe(step,name,state,correction):
                """Record mean state and correction norm in original CSV units."""
                traces.append(dict(training_seed=training_seed,seed=seed,method=name,step=step,
                    mean_consumption=float(state.mean())*scale,
                    correction_l2=float(torch.linalg.vector_norm(correction,dim=1).mean())*scale))
            for name in METHODS:
                observe(0,name,noise,torch.zeros_like(noise))
            samples = sample_schedules(model,noise,ramp/scale,schedules,trace=observe)
            saved = {name:x.numpy()*scale for name,x in samples.items()}
            # D must remain the projection of A, regardless of schedule batching.
            np.testing.assert_allclose(samples["D"].numpy(),project_profiles(samples["A"],ramp/scale).numpy(),rtol=0,atol=2e-6)
            for name,x in saved.items():
                rows.append(dict(training_seed=training_seed,seed=seed,method=name,
                    projection_calls=len(schedules[name]),mean_consumption=float(x.mean()),
                    **metrics(x,reference,ramp,directions,config["feasibility_tolerance_csv"])))
            np.savez_compressed(out / f"samples_training_{training_seed}_seed_{seed}.npz",reference=reference,**saved)
            print(f"Completed training seed {training_seed}, sampling seed {seed}",flush=True)
    frame, trajectory = pd.DataFrame(rows), pd.DataFrame(traces)
    frame.to_csv(out / "metrics.csv",index=False)
    trajectory.to_csv(out / "trajectories.csv",index=False)
    pd.DataFrame(baseline).to_csv(out / "reference_metrics.csv",index=False)
    keys = ["feasible_fraction","sliced_w1","daily_total_w1","mean_consumption"]
    frame.groupby(["training_seed","method"])[keys].agg(["mean","std"]).to_csv(out / "within_training_summary.csv")
    per_model = frame.groupby(["training_seed","method"])[keys].mean()
    per_model.to_csv(out / "per_training_seed.csv")
    per_model.groupby("method").agg(["mean","std"]).to_csv(out / "between_training_summary.csv")
    paired = paired_differences(frame)
    paired.to_csv(out / "paired_late_minus_early.csv")
    paired.groupby("training_seed").agg(["mean","std"]).to_csv(out / "paired_summary.csv")
    plot_schedules(out,frame,trajectory)
    if len(frame) != 150 or frame.groupby(["training_seed","seed"]).method.nunique().ne(5).any():
        raise RuntimeError("Incomplete schedule comparison")
    (out / "COMPLETE.json").write_text(json.dumps(dict(status="complete",rows=len(frame)),indent=2)+"\n")
    # Publishing the pointer last prevents notebooks from loading partial runs.
    manifest = ROOT / "schedule_run.json"
    temp = manifest.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(run_dir=str(out.relative_to(ROOT))),indent=2)+"\n")
    temp.replace(manifest)
    print("COMPLETE",out,flush=True)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
