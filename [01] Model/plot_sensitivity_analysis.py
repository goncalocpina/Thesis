"""
Author: Gonçalo Costa Pina
Sensitivity plot — Total System Cost vs WACC and Lifetime
(relative change from baseline)

Run from the [01] Model folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Thesis styling ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "font.family":        "serif",
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "lines.linewidth":    1.2,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.format":     "pdf",
})

# ── Baselines ─────────────────────────────────────────────────────────────────
BASELINE_LIFETIME = 20
BASELINE_WACC     = 0.083616
BASELINE_MARGIN   = 8.2/100

# ── Load data ─────────────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "Logbooks" / "Logbook_FINAL_MODEL_WACC_LIFETIME.csv"
df = pd.read_csv(csv_path)

col_wacc     = "parameters.wacc"
col_lifetime = "parameters.general_lifetime"
col_margin   = "parameters.margin"

#NOTE: two options for y-axis
#col_cost = "financial.total_system_cost_eur_per_yr"
col_cost = "financial.npv_final_eur"

# ── Separate the three sweeps ─────────────────────────────────────────────────
# WACC sweep: lifetime and margin held at baseline
df_wacc = df[
    np.isclose(df[col_lifetime], BASELINE_LIFETIME) &
    np.isclose(df[col_margin],   BASELINE_MARGIN)
].copy()
df_wacc = df_wacc.sort_values(col_wacc)
df_wacc["x_rel"] = (df_wacc[col_wacc] - BASELINE_WACC) / BASELINE_WACC * 100  # %

# Lifetime sweep: WACC and margin held at baseline
df_lt = df[
    np.isclose(df[col_wacc],   BASELINE_WACC) &
    np.isclose(df[col_margin], BASELINE_MARGIN)
].copy()
df_lt = df_lt.sort_values(col_lifetime)
df_lt["x_rel"] = (df_lt[col_lifetime] - BASELINE_LIFETIME) / BASELINE_LIFETIME * 100  # %

# Margin sweep: WACC and lifetime held at baseline
df_margin = df[
    np.isclose(df[col_wacc],      BASELINE_WACC) &
    np.isclose(df[col_lifetime],  BASELINE_LIFETIME)
].copy()
df_margin = df_margin.sort_values(col_margin)
df_margin["x_rel"] = (df_margin[col_margin] - BASELINE_MARGIN) / BASELINE_MARGIN * 100  # %

# ── Normalise cost to baseline ────────────────────────────────────────────────
baseline_cost = df[
    np.isclose(df[col_wacc],     BASELINE_WACC) &
    np.isclose(df[col_lifetime], BASELINE_LIFETIME) &
    np.isclose(df[col_margin],   BASELINE_MARGIN)
][col_cost].values[0]

df_wacc["cost_norm"]   = (df_wacc[col_cost]   / baseline_cost - 1) * 100
df_lt["cost_norm"]     = (df_lt[col_cost]     / baseline_cost - 1) * 100
df_margin["cost_norm"] = (df_margin[col_cost] / baseline_cost - 1) * 100

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,4))

# WACC sweep
ax.plot(
    df_wacc["x_rel"],
    df_wacc["cost_norm"],
    color="steelblue",
    marker="x",
    linestyle="--",
    markersize=5,
    label=f"WACC  (baseline: {BASELINE_WACC*100:.2f}%)",
    zorder=3,
)

# Lifetime sweep
ax.plot(
    df_lt["x_rel"],
    df_lt["cost_norm"],
    color="firebrick",
    marker="s",
    linestyle=":",
    markersize=5,
    label=f"Lifetime  (baseline: {BASELINE_LIFETIME} yr)",
    zorder=3,
)

# Margin sweep
ax.plot(
    df_margin["x_rel"],
    df_margin["cost_norm"],
    color="seagreen",
    marker="o",
    linestyle="-.",
    markersize=5,
    label=f"Margin  (baseline: {BASELINE_MARGIN*100:.1f}%)",
    zorder=3,
)

# Baseline marker
ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5, label="Baseline")
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
#ax.plot(0, 0, "k*", markersize=10, zorder=5)

# Axis labels
ax.set_xlabel(r"Change from baseline / [\%]")
ax.set_ylabel(r"Change in NPV / [\%]")

ax.legend(fontsize=9)
#ax.set_title("Sensitivity of total system cost to WACC and project lifetime\n"
#             f"Baseline: WACC = {BASELINE_WACC*100:.2f}%, Lifetime = {BASELINE_LIFETIME} yr, "
#             f"Margin = {BASELINE_MARGIN*100:.1f}%")

plt.tight_layout()

output = Path(__file__).parent.parent / "[03] Figures" / "Sensitivity_WACC_Lifetime" / "SENSITIVITY_cost_vs_wacc_lifetime.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")

plt.show()