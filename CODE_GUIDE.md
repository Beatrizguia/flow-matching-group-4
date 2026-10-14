# Code guide

## Execution and data flow

1. `prepare_dataset.py` verifies/downloads the real hourly CSV. It never fabricates data or substitutes the synthetic files in the same record.
2. `execute_project.py` executes each notebook in a fresh kernel using the active Python interpreter and the notebook directory. The notebooks explain the experiment in English, load or train models, and produce tables and plots.
3. `torus_run.json` and `load_run.json` select the documented primary experiments. `supplementary_run.json` selects the completed extension. `schedule_run.json` selects the completed fixed-count timing study. Paths are relative to the project root.

## Notebook presentation: `notebook_utils.py`

Shared table formatting shows mean ± sample SD without altering measurements. Percentage conversion is display-only; small constraint residuals retain scientific notation. The Colab setup cells locate the complete project before importing modules. `configure_plots` assigns consistent colours to A–D across both domains.

## Torus: `torus_project.py`

`sample_torus` generates training points uniform in surface area, correcting the poloidal-angle density by rejection sampling. `FlowMatchingNetwork` predicts 3D velocity from a point and scalar flow time. Training is implemented explicitly in the notebook.

`torus_h` returns a quadratic constraint residual, not surface distance. `hard_project` finds the nearest torus point through a radial/tube construction. At the axis or tube centre, deterministic conventions resolve nonunique choices. `soft_correct` implements the prescribed normalized equality correction.

`sample_abcd` uses three paired trajectories: ordinary Euler for A, soft-corrected Euler for B, projected Euler for C. D reuses A's endpoint and projects it once. The input noise remains unchanged.

`reference_torus`, `sw1`, `measure`, `torus_angles`, `phi_cdf` and `phi_ks` implement independent reference sampling and feasibility/fidelity metrics. `supplementary_analysis` derives all-seed angular metrics and paired SW1 differences from saved samples. It does not fit a model.

## Residential: `load_profile_project.py`

`prepare_data` checks consecutive, unique UTC timestamps and full days, reshapes data to household/day/hour, and removes nonfinite or negative days. It splits household identities before flattening days. It returns original-unit float32 arrays of shape `(n_days, 24)` and provenance metadata. The scale and empirical ramp come only from training data.

`Velocity` maps `(batch, 24)` plus `(batch, 1)` flow times to velocities of shape `(batch, 24)`. Training/sampling use `y=x/scale`, without centering; evaluation converts back to CSV units. Flow time from 0 to 1 is different from the 24 clock hours in each profile.

`train_model` resamples independent Gaussian noise, target profiles and uniform interpolation times at each iteration. The target is `x1-x0`; MSE trains the network along straight interpolation paths. A fixed validation regression set selects the checkpoint. Test data are never used for this selection. The saved residential checkpoints support inference, not exact optimizer/RNG resumption.

`constraint_matrix` represents 24 nonnegative bounds and 46 signed adjacent-hour ramp inequalities. There is no midnight wraparound. `soft_correction` takes one gradient step on the squared hinge penalty with fixed divisor 9. This adapts B to inequalities; it differs from the torus equality correction.

`project_profiles` uses Dykstra correction vectors to project onto the intersection of the nonnegative orthant and two disjoint-pair ramp-slab sets. It checks both iterate change and feasibility and fails on nonconvergence. Simple clipping until feasible would not establish nearest-point optimality. Tests compare with an independent SLSQP solution.

`sample_four` shares weights/noise across A-D and optionally calls a read-only observer after each update. D follows A until a separate terminal event. `metrics` distinguishes full-profile feasibility, component violation frequencies/magnitudes, SW1, mean/variability errors, daily-total and peak W1, and within-day autocorrelation error. `evaluate` uses disjoint test reference subsets within each seed; `plot_results` illustrates seed 0 and displays all-seed summaries.

## Extension: `supplementary_experiments.py`

`main` runs nonnegative-only, ramp-only and joint ablations on validation references; records intermediate trajectories; trains seeds 43/44; and combines them with the documented seed 42. `paired_table` subtracts A within each seed. `plot_supplement` visualizes the recorded numerical diagnostics.

`active_feasible_fraction` checks only the ablation's active bounds; `feasible_fraction` always checks the original joint bounds. A ramp-only sample can still be negative. The soft divisor remains 9 when constraints are removed, isolating the constraint-set change. Training robustness first averages ten sampling seeds within each fitted model, then summarizes three model-level means. Thirty observations are not thirty independent trainings.

## Fixed-count timing: `schedule_experiments.py`

`projection_schedules` explicitly records one-based post-Euler steps. Early25
uses 1–24 plus 100; Late25 uses 76–100. Each includes the final projection and
makes 25 projection calls. Their contrast changes placement at fixed count;
comparisons against C (100 calls) or D (one call) do not hold count fixed.

`sample_schedules` batches the trajectories through the unchanged velocity
field, validates schedule indices, preserves initial noise and uses the same
Dykstra operator. `paired_differences` subtracts Early25 from Late25 separately
for each trained model and sampling seed. Negative SW1 differences favour late
placement. Summary SDs distinguish ten within-model observations from three
model-level means. `plot_schedules` only displays recorded values.

The runner checks dataset/split/scale and checkpoint provenance, writes its
protocol before sampling, saves every sample set, and updates the manifest
only after completion. It uses validation references and never retrains a
model. Frozen source snapshots document the code used for the recorded run;
subsequent formatting or utility fixes do not rewrite historical snapshots.

## Result records and tests

Each primary run includes `samples_seed_*.npz` with reference/A/B/C/D arrays, per-seed `metrics.csv`, configuration and checkpoints. Ten seeds each contain 2,000 samples. Supplemental records include per-constraint metrics, trajectories, source hashes, two additional checkpoints and model-level summaries. Historical source snapshots remain immutable.

`test_torus.py` checks singular projections, analytic gradients, known Euler trajectories, saved samples, analytic CDF and KS against SciPy. `test_project.py` checks residential projection optimality, soft descent, pairing, splitting and metric identities. `test_extensions.py` checks all constraint subsets, mean preservation, nonintrusive tracing and malformed-input rejection. `test_experiments.py` checks paired contrasts, complete A-D seed tables, identical ablation baselines and correct model-level aggregation.

All active Python modules contain English docstrings explaining mathematical assumptions, tensor shapes and side effects. The notebooks provide the interpretation; the modules provide the shared implementation used by both notebooks and tests.

`test_schedules.py` verifies projection placement against an analytic time-dependent
field, matches the established A/C/D sampler, rejects invalid schedules and
checks nonintrusive tracing. With a completed run it also checks source and
checkpoint hashes, all 150 metric rows against saved samples, paired differences
and model-level means. `test_extensions.py` additionally covers evaluation of a
seed subset without seed 0; measurements are returned with no plot illustration.
