"""Regression checks for paired comparisons and completed experiment tables.

Run without Python's -O flag after the supplementary experiment exists.
Check paired differences using known synthetic values, then verify complete
real experiment tables and correct model-level aggregation.
The script reads the saved runs; it does not retrain models or overwrite data.
"""

import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
from supplementary_experiments import paired_table

# Use a known paired contrast to verify the helper independently of real data.
frame = pd.DataFrame(
    {
        "seed": [0, 0, 0, 0, 1, 1, 1, 1],
        "method": list("ABCD") * 2,
        "sliced_w1": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
    }
)
frame.loc[frame.method == "A", "sliced_w1"] += 10
paired = paired_table(frame)
assert (paired["B-A"] == -9).all()
extra = ROOT / json.loads((ROOT / "supplementary_run.json").read_text())["run_dir"]
a = pd.read_csv(extra / "ablation_metrics.csv")
m = pd.read_csv(extra / "training_seed_metrics.csv")
assert len(a) == 120 and len(m) == 120
assert a.groupby(["constraints", "seed"]).method.nunique().eq(4).all()
assert m.groupby(["training_seed", "seed"]).method.nunique().eq(4).all()
assert set(m.training_seed) == {42, 43, 44}
assert set(m.seed) == set(range(10))
# All ablations share the exact A baseline for a given sampling seed.
assert (
    a.groupby("seed")
    .apply(
        lambda g: g[g.method == "A"].sliced_w1.max()
        - g[g.method == "A"].sliced_w1.min(),
        include_groups=False,
    )
    .max()
    < 1e-12
)

# Reconstruct model-level means to catch accidental pooling of all 30 runs.
p = pd.read_csv(extra / "per_training_seed.csv")
computed = (
    m.groupby(["training_seed", "method"])[
        ["feasible_fraction", "sliced_w1", "daily_total_w1"]
    ]
    .mean()
    .reset_index()
)
pd.testing.assert_frame_equal(p, computed, check_exact=False, atol=1e-12, rtol=1e-12)
print(
    "PASS: paired deltas, complete ablation and correct model-level aggregation."
)
