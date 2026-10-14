"""Run the fixed residential extension and expose paired summary helpers.

The extension is descriptive follow-up, not hyperparameter optimization:
validation references support the constraint ablation; fixed test references
support three-training-seed robustness. Main torus settings are untouched.
Running this file writes a new timestamped run and updates its manifest only
after completion. Importing helpers does not run experiments or select a
Matplotlib backend. Historical source snapshots remain immutable evidence.
"""

from pathlib import Path
from datetime import datetime
import json, hashlib, platform, importlib.metadata, shutil
import numpy as np
import pandas as pd
import torch
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_profile_project import (
    Velocity,
    prepare_data,
    train_model,
    evaluate,
    sample_four,
    constraint_matrix,
    metrics,
)

ROOT = Path(__file__).resolve().parent


def paired_table(frame, metric="sliced_w1"):
    """Return per-seed B-A, C-A and D-A differences for the selected metric.

    frame must contain one row per (seed, method) with all A/B/C/D methods.
    Positive SW1 differences mean worse fidelity than A. Pair before summarizing
    to retain the shared-reference/noise design; do not subtract unpaired SDs.
    """
    pivot = frame.pivot(index="seed", columns="method", values=metric)
    return pd.DataFrame({m + "-A": pivot[m] - pivot["A"] for m in "BCD"})


def load_model(run):
    """Load a trusted residential run's model.pt onto CPU in evaluation mode.

    run is a Path to a project-produced checkpoint directory. This loader uses
    weights_only=False for that trusted artifact; do not pass arbitrary files.
    No fitting or parameter selection occurs here.
    """
    cp = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    model = Velocity()
    model.load_state_dict(cp["model_state_dict"])
    return model.eval()


def main():
    """Execute ablation, trajectory diagnostics and two additional trainings.

    Reuse the documented seed-42 model; train seeds 43 and 44 independently.
    Write the protocol and source hashes first, retain per-seed measurements,
    and average within each model before reporting between-training SD. No
    ablation winner is promoted to replace the primary experiment. Files are
    written under outputs/supplementary; the completion manifest is written last.
    """
    torch.set_num_threads(4)
    arrays, info = prepare_data(ROOT / "fluvius_wide_format.csv")
    base = ROOT / json.loads((ROOT / "load_run.json").read_text())["run_dir"]
    out = (
        ROOT
        / "outputs"
        / "supplementary"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True)
    # Written before experiments: fixed settings, no winner selection.
    config = dict(
        training_seeds=[42, 43, 44],
        sampling_seeds=list(range(10)),
        training_iterations=5000,
        n_samples=2000,
        n_steps=100,
        ablation_split="validation",
        robustness_split="test",
        ablation_model_training_seed=42,
        constraints=["nonnegative", "ramp", "both"],
        soft_strength=0.3,
        soft_divisor=9,
        sw_seed=2026,
        sw_directions=128,
        selection="No tuning; descriptive follow-up after historical test inspection",
        base_run=str(base.relative_to(ROOT)),
        data_sha256=info["sha256"],
        versions={
            p: importlib.metadata.version(p)
            for p in ["torch", "numpy", "pandas", "scipy", "matplotlib"]
        },
        python=platform.python_version(),
        code_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in ["load_profile_project.py", "supplementary_experiments.py"]
        },
    )
    (out / "protocol.json").write_text(json.dumps(config, indent=2))
    (out / "source").mkdir()
    for name in config["code_sha256"]:
        shutil.copyfile(ROOT / name, out / "source" / name)
    (out / "data_config.json").write_text(json.dumps(info, indent=2))
    checkpoint = torch.load(base / "model.pt", map_location="cpu", weights_only=False)
    recorded = json.loads((base / "data_config.json").read_text())
    if (
        checkpoint["seed"] != 42
        or checkpoint["scale"] != info["scale"]
        or recorded["sha256"] != info["sha256"]
    ):
        raise ValueError(
            "Extension requires the documented seed-42 checkpoint and matching dataset/scale"
        )
    model = load_model(base)
    scale, ramp = info["scale"], info["ramp_threshold"]
    dirs = np.random.default_rng(2026).normal(size=(24, 128))
    dirs /= np.linalg.norm(dirs, axis=0)
    rows, traces = [], []
    for constraints in config["constraints"]:
        for seed in config["sampling_seeds"]:
            # Recreate the same seed inside every constraint condition,
            # pairing the ablations rather than comparing unrelated noise.
            noise = torch.randn(2000, 24, generator=torch.Generator().manual_seed(seed))
            idx = np.random.default_rng(10000 + seed).choice(
                len(arrays["validation"]), 2000, replace=False
            )
            ref = arrays["validation"][idx]

            def record(t, method, state, correction, event):
                """Record diagnostics in original units without modifying sampling tensors.

                This synchronous callback captures the current constraint set and seed.
                Average violation is over active SIGNED inequalities; mean correction L2 is
                a profile-level norm. Neither is the fraction of fully feasible profiles.
                """
                G, b = constraint_matrix(
                    ramp / scale, state.device, state.dtype, constraints
                )
                traces.append(
                    dict(
                        constraints=constraints,
                        seed=seed,
                        time=t,
                        method=method,
                        event=event,
                        mean_consumption=float(state.mean()) * scale,
                        mean_active_violation=float(torch.relu(state @ G.T - b).mean())
                        * scale,
                        correction_l2=float(
                            torch.linalg.vector_norm(correction, dim=1).mean()
                        )
                        * scale,
                    )
                )

            for method in "ABCD":
                record(0.0, method, noise, torch.zeros_like(noise), "initial")
            samples = sample_four(
                model, noise, ramp / scale, constraints=constraints, trace=record
            )
            saved = {}
            for method, value in samples.items():
                x = value.numpy() * scale
                saved[method] = x
                m = metrics(x, ref, ramp, dirs, max(1e-5, scale * 1e-5))
                G, b = constraint_matrix(ramp, "cpu", torch.float32, constraints)
                m["active_feasible_fraction"] = float(
                    (torch.relu(torch.from_numpy(x) @ G.T - b).max(1).values <= 1e-5)
                    .float()
                    .mean()
                )
                m["mean_consumption"] = float(x.mean())
                rows.append(
                    dict(constraints=constraints, seed=seed, method=method, **m)
                )
            if seed == 0:
                np.savez_compressed(
                    out / f"ablation_{constraints}_seed_0.npz", reference=ref, **saved
                )
        print("Finished validation ablation:", constraints, flush=True)
    ablation = pd.DataFrame(rows)
    ablation.to_csv(out / "ablation_metrics.csv", index=False)
    ablation.groupby(["constraints", "method"])[
        [
            "active_feasible_fraction",
            "feasible_fraction",
            "sliced_w1",
            "mean_consumption",
        ]
    ].agg(["mean", "std"]).to_csv(out / "ablation_summary.csv")
    trace = pd.DataFrame(traces)
    trace.to_csv(out / "trajectories.csv", index=False)
    history_rows = []
    # Training interpolation diagnostic uses the same normalized Gaussian and real training endpoints.
    rng = np.random.default_rng(9001)
    endpoint = torch.from_numpy(
        arrays["train"][rng.choice(len(arrays["train"]), 2000, replace=False)] / scale
    )
    source = torch.randn(2000, 24, generator=torch.Generator().manual_seed(9001))
    G, b = constraint_matrix(ramp / scale, "cpu", torch.float32)
    for t in np.linspace(0, 1, 101):
        x = (1 - t) * source + t * endpoint
        history_rows.append(
            dict(
                time=t,
                mean_consumption=float(x.mean()) * scale,
                mean_active_violation=float(torch.relu(x @ G.T - b).mean()) * scale,
            )
        )
    pd.DataFrame(history_rows).to_csv(out / "training_path_diagnostic.csv", index=False)
    plot_supplement(out, ablation, trace, pd.DataFrame(history_rows))
    # Three training seeds, each summarized over ten common sampling/reference seeds.
    all_frames = []
    historical = pd.read_csv(base / "metrics.csv")
    historical["training_seed"] = 42
    all_frames.append(historical)
    for seed in [43, 44]:
        run = out / f"training_seed_{seed}"
        run.mkdir()
        trained, history, selected = train_model(
            arrays["train"], arrays["validation"], scale, run, seed=seed
        )
        frame, _, _ = evaluate(trained, arrays, info, run)
        frame["training_seed"] = seed
        all_frames.append(frame)
        (run / "config.json").write_text(
            json.dumps(
                dict(
                    training_seed=seed,
                    selected_step=selected,
                    protocol="../protocol.json",
                ),
                indent=2,
            )
        )
    multi = pd.concat(all_frames, ignore_index=True)
    multi.to_csv(out / "training_seed_metrics.csv", index=False)
    # A fitted model is the replication unit for training uncertainty.
    # Average its ten sampling runs first; only then compute across-model SD.
    per_training = multi.groupby(["training_seed", "method"])[
        ["feasible_fraction", "sliced_w1", "daily_total_w1"]
    ].mean()
    per_training.to_csv(out / "per_training_seed.csv")
    per_training.groupby("method").agg(["mean", "std"]).to_csv(
        out / "between_training_summary.csv"
    )
    for name, frame in [("historical", historical)]:
        paired = paired_table(frame)
        paired.to_csv(out / f"{name}_paired_sw1.csv")
        paired.agg(["mean", "std"]).T.to_csv(out / f"{name}_paired_sw1_summary.csv")
    paired = multi.pivot(
        index=["training_seed", "seed"], columns="method", values="sliced_w1"
    )
    delta = pd.DataFrame({m + "-A": paired[m] - paired.A for m in "BCD"})
    delta.to_csv(out / "paired_sw1_by_training_seed.csv")
    delta.groupby("training_seed").mean().to_csv(out / "paired_training_means.csv")
    plot_training_robustness(out, per_training)
    (out / "COMPLETE.json").write_text(
        json.dumps({"status": "complete", "rows": len(multi)}, indent=2)
    )
    (ROOT / "supplementary_run.json").write_text(
        json.dumps({"run_dir": str(out.relative_to(ROOT))}, indent=2) + "\n"
    )
    print("COMPLETE", out, flush=True)


def plot_training_robustness(out, per_training):
    """Save model-level mean comparisons with consistent A–D colours.

    per_training is indexed by training_seed and method; each row already
    averages the ten sampling seeds. No metrics or uncertainty are recomputed.
    """
    from notebook_utils import COLORS

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    means = per_training.reset_index()
    for method, g in means.groupby("method"):
        axes[0].plot(g.training_seed, g.sliced_w1, "o-", label=method, color=COLORS[method])
        axes[1].plot(g.training_seed, g.feasible_fraction, "o-", label=method, color=COLORS[method])
    axes[0].set(ylabel="SW1 (CSV units)")
    axes[1].set(ylabel="Feasible fraction")
    for ax in axes:
        ax.set(xlabel="Training seed", xticks=[42, 43, 44])
        ax.legend()
    fig.suptitle("Each point averages 10 sampling seeds; 3 trained models")
    fig.tight_layout()
    fig.savefig(out / "training_robustness.png", dpi=160)
    plt.close(fig)


def plot_supplement(out, ablation, trace, interpolation):
    """Save trajectory and validation-ablation figures from computed tables.

    Trajectory lines average ten sampling seeds; the interpolation curve is a
    separate training-path diagnostic. Preserve event order so D's terminal
    jump at t=1 remains distinct. Ablation bars report sample SD and distinguish
    active-set feasibility from feasibility under both original constraints.
    """
    from notebook_utils import COLORS

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    selected = trace[trace.constraints == "both"]
    for method in "ABCD":
        g = (
            selected[selected.method == method]
            .groupby(["time", "event"], sort=False)[
                ["mean_consumption", "mean_active_violation", "correction_l2"]
            ]
            .mean()
            .reset_index()
        )
        for ax, key in zip(
            axes, ["mean_consumption", "mean_active_violation", "correction_l2"]
        ):
            ax.plot(g.time, g[key], label=method, alpha=0.85, color=COLORS[method])
    for ax, key in zip(axes[:2], ["mean_consumption", "mean_active_violation"]):
        ax.plot(
            interpolation.time,
            interpolation[key],
            "k--",
            label="Training interpolation",
        )
    for ax, label in zip(
        axes, ["Mean consumption", "Mean active violation", "Mean correction L2"]
    ):
        ax.set(xlabel="Flow time", ylabel=label + " (CSV units)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(
        "Both constraints: means over 10 sampling seeds; D projects only at t=1"
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, key, title in zip(
        axes,
        ["sliced_w1", "active_feasible_fraction", "feasible_fraction"],
        [
            "Validation SW1",
            "Feasibility: active constraints",
            "Feasibility: both constraints",
        ],
    ):
        grouped = ablation.groupby(["constraints", "method"])[key].agg(["mean", "std"])
        for i, c in enumerate(["nonnegative", "ramp", "both"]):
            g = grouped.loc[c].reindex(list("ABCD"))
            ax.bar(
                np.arange(4) + (i - 1) * 0.25,
                g["mean"],
                0.25,
                yerr=g["std"],
                capsize=2,
                label=c,
            )
        ax.set(xticks=np.arange(4), xticklabels=list("ABCD"), title=title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(
        "Fixed model (training seed 42), 10 paired sampling seeds; mean +/- SD"
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(out / "ablation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
