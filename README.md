# Constraint timing in Flow Matching

**DM2583 — Group 4**  
Maria do Mar Vieira, Beatriz Guia and Renata Parmanova.

This code accompanies the final technical report, supplied separately. It compares unconstrained sampling (A), soft correction (B), projection after every Euler step (C), and terminal projection (D) on a surface-uniform torus and 24-hour residential consumption profiles.

All strategies share model weights and initial noise within each comparison.

## Dataset download

The dataset file `fluvius_wide_format.csv` (approximately 89 MB) is **not included in this GitHub repository** because of its size.

Download it from the original MODERATE dataset on Zenodo:

- [Dataset page](https://zenodo.org/records/20525554)
- [Download fluvius_wide_format.csv](https://zenodo.org/records/20525554/files/fluvius_wide_format.csv?download=1)

Place `fluvius_wide_format.csv` in the project root, alongside `flow_matching.ipynb` and `prepare_dataset.py`.

Download the real `fluvius_wide_format.csv` file, rather than the synthetic profile files available on the same Zenodo page.

Alternatively, download and verify the dataset automatically by running this command from the project directory:

```bash
python prepare_dataset.py
```

The script verifies the SHA-256 checksum and places the file beside the notebooks. It reuses a valid existing file and rejects a mismatching file without overwriting it.

Expected SHA-256:

```text
bf490309adedd05549f90c97f6b44802f812eedf0f21bef4d0f5bafd3229c7b9
```

The dataset contains 8,760 UTC hours, 1,300 households and 474,500 complete household-days. The split uses 910 training, 195 validation and 195 test households, with no household shared across partitions.

Scaling and the ramp threshold use training data only. Consumption is reported in original CSV units because the source's physical-unit and aggregation documentation is incomplete.

Nonnegativity is an operational assumption for this nonnegative consumption variable. The ramp limit is an empirical restriction, not a physical law. Approximately 7.94% of real test days violate it and remain in the fidelity reference.

## Open in Google Colab

Upload the complete project folder to Google Drive as:

```text
My Drive/flow_matching_project
```

Open `toymodel.ipynb` first, then `flow_matching.ipynb`, with Google Colab.

Choose **Runtime → Run all** and authorise the Drive mount when requested. Keep `TRAIN_FROM_SCRATCH=False` to analyse the documented results.

The first setup cell finds the folder. Set `PROJECT_DIRECTORY` if you stored it elsewhere. Uploading only a notebook does not provide its modules, checkpoints or result files.

A CPU runtime is sufficient for saved-result analysis. Missing analysis packages are installed only in Colab, while existing versions are preserved.

The residential notebook verifies the CSV and downloads the documented real file if it is absent. You can also download it manually using the links above.

Both notebooks include saved outputs for reading without execution.

## Run the documented experiments locally

The original experiments used Python 3.11.9. The final implementation was checked with Python 3.13.7.

Run these commands from the project directory on macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python prepare_dataset.py
python execute_project.py toymodel.ipynb flow_matching.ipynb
```

On Windows, run:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-audit.txt
.venv\Scripts\python.exe prepare_dataset.py
.venv\Scripts\python.exe execute_project.py toymodel.ipynb flow_matching.ipynb
```

`requirements.txt` records historical versions. `requirements-audit.txt` records the tested update environment. Bitwise equality across platforms is not promised.

Both notebooks default to `TRAIN_FROM_SCRATCH=False`. Run All loads saved checkpoints, samples and metric tables selected by the four `*_run.json` manifests, then regenerates analysis.

The notebook executor uses each notebook's directory as its kernel working directory, even when launched elsewhere. The residential notebook requires the CSV for provenance checks and exploratory analysis.

## Train new models

Set `TRAIN_FROM_SCRATCH=True` in the relevant notebook.

New runs go into timestamped directories without replacing the documented manifests. Update a manifest deliberately only after a successful run.

The final report describes the supplied historical runs. Changing the manifests does not update that report automatically.

## Supplementary experiments

Run:

```bash
python supplementary_experiments.py
```

This command reuses the selected seed-42 residential checkpoint, runs the validation constraint ablation and trajectory diagnostics, and trains and evaluates seeds 43 and 44.

It writes a protocol and source snapshots before execution and updates `supplementary_run.json` after completion. It requires the documented seed-42 data and scale.

The notebook skips supplemental results from a different primary run.

The extension followed inspection of the historical test results. It is a descriptive follow-up, not a preregistered untouched-test study.

## Early and late projection schedules

Run:

```bash
python schedule_experiments.py
```

This completed follow-up reuses training seeds 42, 43 and 44, with ten paired sampling seeds per model on validation references.

- **Early25:** projection after steps 1–24 and 100.
- **Late25:** projection after steps 76–100.

Both schedules make 25 projection calls and include the endpoint. A, C and D provide comparison anchors.

There are 2,000 profiles per schedule and sampling seed, with 100 Euler steps. No retraining or schedule tuning is performed.

`schedule_run.json` selects the completed run under `outputs/schedules/`.

Saved files include all 30 sample sets, per-seed measurements, paired Late25-minus-Early25 contrasts, within-model sampling standard deviations, model-level means, trajectory diagnostics, figures, and a protocol with checkpoint and source hashes.

The final section of `flow_matching.ipynb` displays these results, including saved outputs visible without execution.

To regenerate the measurements, run the command above. It writes a fresh run and publishes its manifest last.

Late25 has substantially lower SW1 than Early25 in all three fitted models, while both attain 100% numerical feasibility in all evaluated seeds.

This supports a placement effect at a fixed projection count in the tested setting, not a universally optimal schedule.

Validation reference samples differ from the primary test experiment and the previous ablation, so compare methods within a run. This is an exploratory follow-up after historical test inspection.

## Validate

Run:

```bash
python test_torus.py
python test_project.py
python test_extensions.py
python test_experiments.py
python test_schedules.py
```

On Windows, use `.venv\Scripts\python.exe` instead of `python` if the virtual environment is not activated.

Do not use Python's `-O` flag: these scripts use assertions.

Tests check geometry against independent optimizers, paired trajectories and metrics, household splitting, constraint subsets, input guards and experiment completeness.

The full CSV is needed by `test_project.py`. No model is trained by these tests.

## Project files

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

Frozen `outputs/supplementary/*/source/` files are the exact source used for the recorded extension. Their hashes match its protocol. They document scientific provenance.

The final code package contains no report generator, prose template or checkpoint-document drafts.

## Interpretation and limitations

Sampling standard deviations across ten seeds and training standard deviations across three model-level means measure different sources of variability.

The torus experiment has one training run.

No formal privacy guarantee or universal ranking of constrained generation methods is claimed.
