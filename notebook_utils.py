"""Presentation helpers shared by the two analysis notebooks.

These functions only format copies of result tables and configure plots. They do
not modify samples, metrics, training, or the experiment selection manifests.
"""
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

COLORS = {"Reference": "#505965", "reference": "#505965",
          "A": "#2878B5", "B": "#DE8F05", "C": "#C44E65", "D": "#239B87"}


def configure_plots():
    """Use readable, consistent figures in Jupyter and Google Colab."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "figure.dpi": 95, "savefig.dpi": 160,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=[COLORS[m] for m in "ABCD"]),
        "legend.frameon": False, "figure.facecolor": "white"})


def show_table(table, caption, precision=4):
    """Display a compact HTML table without changing its underlying values."""
    styled = table.style.format(precision=precision, na_rep="—").set_caption(caption)
    styled = styled.set_table_styles([
        {"selector": "caption", "props": [("text-align", "left"),
          ("font-weight", "600"), ("padding", "10px 0")]},
        {"selector": "th", "props": [("background-color", "#20384B"),
          ("color", "white"), ("padding", "8px 12px")]},
        {"selector": "td", "props": [("padding", "7px 12px"),
          ("text-align", "right"), ("border-bottom", "1px solid #dbe3e9")]},
    ])
    display(styled)


def _number(value, percentage=False):
    """Format percentages explicitly; keep small nonzero residuals visible."""
    if percentage:
        # Round away from floating-point tie noise, then use decimal half-up.
        return str(Decimal(str(round(float(value) * 100, 10))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP))
    if 0 < abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def show_summary(frame, labels, caption, percentages=(), group="method"):
    """Show mean ± sample SD across rows within each method (ddof=1).

    labels maps metric columns to readable headings. Percentages are converted
    only for display. The caller specifies the experimental unit in the caption.
    """
    stats = frame.groupby(group)[list(labels)].agg(["mean", "std"])
    table = pd.DataFrame(index=stats.index)
    table.index.name = "Method"
    for key, label in labels.items():
        table[label] = [f"{_number(m, key in percentages)} ± {_number(s, key in percentages)}"
                        for m, s in zip(stats[key]["mean"], stats[key]["std"])]
    show_table(table, caption)
