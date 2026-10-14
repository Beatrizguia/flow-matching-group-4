# Constraint timing in Flow Matching

DM2583, Group 4: Maria do Mar Vieira, Beatriz Guia and Renata Parmanova.

This code accompanies the final technical report, supplied separately. It compares unconstrained sampling (A), soft correction (B), projection after every Euler step (C), and terminal projection (D) on a surface-uniform torus and 24-hour residential consumption profiles. All strategies share model weights and initial noise within each comparison.

## Open in Google Colab

Upload this complete folder to Google Drive as `My Drive/flow_matching_project`.
Open `toymodel.ipynb` first, then `flow_matching.ipynb`, with Google Colab. Choose
**Runtime → Run all** and authorise the Drive mount when requested. Keep
`TRAIN_FROM_SCRATCH=False` to analyse the documented results. The first setup cell
finds the folder; set `PROJECT_DIRECTORY` if you stored it elsewhere. Uploading
only a notebook does not provide its modules, checkpoints or result files.

A CPU runtime is sufficient for saved-result analysis. Missing analysis packages
are installed only in Colab; existing versions are preserved. The residential
notebook verifies the CSV and downloads the documented real file if absent.
Both notebooks include saved outputs for reading without execution.

## Run the documented experiments

The original experiments used Python 3.11.9; the final implementation was checked with Python 3.13.7. Use the tested environment below. Run these commands from this directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python prepare_dataset.py
python execute_project.py toymodel.ipynb flow_matching.ipynb
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\python.exe` in place of `python`; activation is optional when invoking that executable directly. `requirements.txt` records historical versions; `requirements-audit.txt` records the tested update environment. Bitwise equality across platforms is not promised.

Both notebooks default to `TRAIN_FROM_SCRATCH=False`. Run All loads saved checkpoints, samples and metric tables selected by the four `*_run.json` manifests, then regenerates analysis. The notebook executor uses each notebook's directory as its kernel working directory, even when launched elsewhere. The real notebook requires the CSV for provenance checks and exploratory analysis.

## Dataset

`prepare_dataset.py` downloads the real `fluvius_wide_format.csv` from the [MODERATE Zenodo record](https://zenodo.org/records/20525554), verifies SHA-256, and places it beside the notebooks. This local project includes the approximately 89 MB CSV. If you omit it from a submission archive, the downloader restores it. The downloader reuses a valid existing file and rejects a mismatching file without overwriting it.

SHA-256: `bf490309adedd05549f90c97f6b44802f812eedf0f21bef4d0f5bafd3229c7b9`.

There are 8,760 UTC hours, 1,300 households and 474,500 complete household-days. The split is 910/195/195 disjoint households. Scaling and the ramp threshold use training data only. Consumption is reported in CSV units because the source's physical-unit and aggregation documentation is incomplete. Nonnegativity is an operational assumption for this nonnegative consumption variable. The ramp is an empirical restriction, not a physical law; approximately 7.94% of real test days violate it and remain in the fidelity reference.

## Train new models

Set `TRAIN_FROM_SCRATCH=True` in the relevant notebook. New runs go into timestamped directories without replacing the documented manifests. Update a manifest deliberately only after a successful run. The final report describes the supplied historical runs, so changed manifests do not update that report automatically.

```bash
python supplementary_experiments.py
```

This command reuses the selected seed-42 residential checkpoint, runs the validation constraint ablation and trajectory diagnostics, and trains/evaluates seeds 43 and 44. It writes a protocol and source snapshots before execution and updates `supplementary_run.json` after completion. It requires the documented seed-42 data/scale. The notebook clearly skips supplemental results from a different primary run. The extension followed inspection of the historical test results; it is descriptive follow-up, not a preregistered untouched-test study.

## Compare projection timing at a fixed count

```bash
python schedule_experiments.py
```

This completed follow-up reuses training seeds 42/43/44 and ten paired sampling
seeds per model on **validation** references. Early25 projects after steps
1–24 and 100; Late25 after steps 76–100. Both make 25 calls and include the
endpoint. A, C and D provide anchors. There are 2,000 profiles per schedule/seed
and 100 Euler steps. No retraining or schedule tuning is performed.

`schedule_run.json` selects the completed run under `outputs/schedules/`.
Saved files include all 30 sample sets, per-seed measurements, paired
Late25-minus-Early25 contrasts, within-model sampling SDs, model-level means,
trajectory diagnostics, figures, and a protocol with checkpoint/source hashes.
The final section of `flow_matching.ipynb` displays these results, including
saved outputs visible without execution. To regenerate the measurements,
run the command above; it writes a fresh run and publishes its manifest last.

Late25 has substantially lower SW1 than Early25 in all three fitted models,
while both attain 100% numerical feasibility in all evaluated seeds. This
supports a placement effect at fixed count in the tested setting, not a
universal optimal schedule. Validation reference samples differ from the
primary test experiment and the previous ablation; compare methods within a
run. This is exploratory follow-up after historical test inspection.

## Validate

```bash
python test_torus.py
python test_project.py
python test_extensions.py
python test_experiments.py
python test_schedules.py
```

Do not use Python's `-O` flag: these scripts use assertions. Tests check geometry against independent optimizers, paired trajectories and metrics, household splitting, constraint subsets, input guards and experiment completeness. The full CSV is needed by `test_project.py`. No model is trained by these tests.

## Files

- `toymodel.ipynb` and `torus_project.py`: prescribed torus experiment.
- `flow_matching.ipynb` and `load_profile_project.py`: residential experiment.
- `supplementary_experiments.py`: constraint ablation, trajectories and training robustness.
- `schedule_experiments.py`: fixed-count early/late projection experiment, reusing all three trained models.
- `prepare_dataset.py` and `execute_project.py`: reproducible execution utilities.
- `notebook_utils.py`: shared table formatting and plot styles; no numerical methods.
- `test_*.py`: numerical and experiment checks.
- `outputs/`: saved samples, checkpoints, configurations, metric CSVs and figures.
- `CODE_GUIDE.md`: implementation and result-reading guide.
- `VALIDATION.md`: checks and reproducibility limits.

Frozen `outputs/supplementary/*/source/` files are the exact source used for the recorded extension; their hashes match its protocol. They are scientific provenance, not redundant report-writing files. The final code package contains no report generator, prose template or checkpoint-document drafts.

Sampling SD across ten seeds and training SD across three model-level means measure different uncertainties. The torus has one training run. No formal privacy guarantee or universal ranking of constrained generation methods is claimed.
