# Final technical validation

Date: 13 October 2026.

## Current checks

- `test_torus.py`: PASS. Analytic soft gradient vs autograd, singular/idempotent hard projection, known Euler trajectories, saved D = projection(A), analytic CDF and KS vs SciPy.
- `test_project.py`: PASS. Joint Dykstra projection vs independent SLSQP, soft descent, paired trajectories, disjoint households, training-only ramp and zero-error metric identities.
- `test_extensions.py`: PASS. All three constraint subsets vs SLSQP, ramp-only mean preservation, nonintrusive trajectory observation and malformed-input guards.
- `test_experiments.py`: PASS. Known paired contrasts, complete ablation/robustness tables, shared A ablation baselines and correct aggregation into model-level means.
- The real CSV matches its expected SHA-256; all 474,500 complete days are retained, with 910/195/195 disjoint households.
- Primary metrics were recomputed from all ten saved sample sets in each domain. Maximum discrepancy from saved CSV values was approximately 1.42e-14, due to floating-point serialization.
- Seed-0 inference was repeated from both primary checkpoints. Largest coordinate differences from historical samples were approximately 4.04e-5 (torus) and 8.59e-6 (residential), consistent with the changed computation environment. Bitwise equality is not promised.
- Projection stress checks agreed with independent SLSQP solutions within approximately 7.77e-9 for successful independent solves.
- Both analysis-mode notebooks executed in fresh kernels from outside the project directory. No error outputs remain. The residential notebook skips supplemental results belonging to a different primary experiment.
- Historical checkpoints, saved sample arrays and frozen supplemental source snapshots are preserved. Source hashes recorded in the supplemental protocol match its archived files.

## Fixed-count schedule follow-up (13 October 2026)

- Executed `schedule_experiments.py`: three existing trained models, ten paired
  sampling seeds each, five schedules and 2,000 profiles per schedule/seed.
  All 150 metric rows and all 30 sample sets are saved in the selected run.
- Early25 and Late25 each make exactly 25 projection calls including the
  endpoint. A, C and D are interpretation anchors, not equal-count controls.
- `test_schedules.py`: PASS. Analytic time-dependent fixture verifies the
  actual placement and count; A/C/D match the established sampler; observers
  leave samples unchanged and invalid indices are rejected. Source/checkpoint
  hashes, every saved-sample metric, paired contrasts and model means were checked.
- The four earlier test scripts were rerun successfully. A small evaluation
  utility fix now returns `illustration=None` when a valid seed subset omits
  seed 0, instead of raising an uninitialized-variable error. Its regression
  fixture passed. Primary measurements and trained weights are unchanged.
- Both notebooks executed successfully in fresh local Jupyter kernels, from
  outside the project directory, and contain their saved tables and figures.
  Google Colab itself was not executed during this local check; its existing
  Drive setup and relative project paths are retained.
- Late25 has lower mean validation SW1 in all three trained models. Both
  Early25 and Late25 are fully feasible at the numerical tolerance in every
  evaluated seed. Three trained models are not thirty independent trainings.
- No model was retrained, no schedule was tuned against test results and no
  historical run was overwritten. The new validation follow-up is exploratory.
  Its reference sampling differs from both the main test evaluation and the
  earlier validation ablation, so their raw SW1 values are not interchangeable.

## Previously completed training validation

The September implementation review executed the residential notebook from scratch in an isolated copy, completing 5,000 iterations and ten sampling seeds. The documented extension independently trained seeds 43 and 44 and evaluated ten sampling seeds per model; seed 42 reuses the primary checkpoint. This final review did not retrain these models or select a new winning run. The prescribed torus retains its original trained model and samples.

## Submission cleanup

Report-generation code, the prose template, superseded narrative reports/checklists and old checkpoint-document copies were removed from the code submission. Numerical checks formerly mixed into `test_reporting.py` remain in `test_experiments.py`. README and CODE_GUIDE describe the final files and reproducible commands. Scientific source snapshots remain because they document the actual recorded extension.

The final technical report is supplied separately. The new fixed-count experiment must be described there if included in the submitted code; the Word document is not automatically rewritten by these scripts.

## Scientific and reproducibility limits

The torus has one training run. Residential training robustness uses three model-level means, not thirty independent trainings. The extension followed inspection of historical test results and is descriptive follow-up. The ramp is empirical and excludes about 7.94% of actual test days. Physical units and upstream aggregation of the source CSV remain incompletely documented; nonnegativity is an operational assumption. These limits are disclosed in the report and are not hidden by the numerical feasibility results.

Original torus optimizer/RNG states were not saved; retrospectively recovered metadata does not certify exact historical training replay. Numerical tests do not prove correctness for every conceivable input or environment. This review verifies the documented implementation and deliverables; it cannot certify a grade or historical submission deadlines.
