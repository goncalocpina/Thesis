"""
Author: Gonçalo Costa Pina (adapted)
Component capacity comparison — Remove-component sensitivity cases

Figures produced:
  1. All components — 2×4 subplot bar chart (original)
  2. Supply components (PV + Wind)
  3. PtX components (Electrolyser, Heat pump, Fuel cell)
  4. Storage components (Battery, Salt cavern, Thermal storage)
  5. Composition — 100% stacked bar (supply / PtX / storage shares)
  6. Index chart — each category relative to base scenario (base = 100)

Run from the [01] Model folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from io import StringIO

# ── Thesis styling ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "font.family":        "serif",
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "lines.linewidth":    1.2,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.format":     "pdf",
})

# ── Load data ──────────────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "Logbooks" / "Logbook_FINAL_MODEL_REMOVE_COMPONENTS.csv"

with open(csv_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

max_fields    = max(line.count(",") + 1 for line in lines)
header_fields = lines[0].count(",") + 1
extra         = max_fields - header_fields
padded_header = lines[0].rstrip("\n") + "".join(f",_extra_{i}" for i in range(extra)) + "\n"
df = pd.read_csv(StringIO(padded_header + "".join(lines[1:])), low_memory=False)

# ── Row order ──────────────────────────────────────────────────────────────────
# CSV rows: 0=no fuel cell | 1=no battery | 2=no salt cavern
#           3=base         | 4=no PV      | 5=no wind (infeasible — excluded)
# Plot order: base (3), no PV (4), no fuel cell (0), no battery (1), no salt cavern (2)

ROW_ORDER = [3, 4, 0, 1, 2]

scenario_labels = [
    "Base",
    "No PV",
    "No fuel cell",
    "No battery",
    "No salt cavern",
]

assert len(df) >= max(ROW_ORDER) + 1, (
    f"CSV has {len(df)} rows but ROW_ORDER references row {max(ROW_ORDER)}."
)

n_scenarios = len(scenario_labels)

# ── Component columns (energy-only for storage) ────────────────────────────────
component_cols = {
    "PV":              "capacities.capacity_pv_mw",
    "Wind":            "capacities.capacity_wind_mw",
    "Electrolyser":    "capacities.capacity_electrolyser_mw",
    "Fuel cell":       "capacities.capacity_fuelcell_mw",
    "Heat pump":       "capacities.capacity_heatpump_mw",
    "Battery":         "capacities.capacity_battery_energy_mwh",
    "Salt cavern":     "capacities.capacity_saltcavern_energy_mwh",
    "Thermal storage": "capacities.capacity_thermalstorage_mwh",
}

missing = [col for col in component_cols.values() if col not in df.columns]
if missing:
    print("WARNING — columns not found in CSV (skipped):")
    for m in missing:
        print(f"  {m}")
    component_cols = {k: v for k, v in component_cols.items() if v not in missing}

# ── Component groupings ────────────────────────────────────────────────────────
GROUPS = {
    "Supply":  ["PV", "Wind"],
    "PtX":     ["Electrolyser", "Fuel cell", "Heat pump"],
    "Storage": ["Battery", "Salt cavern", "Thermal storage"],
}

GROUPS = {
    grp: [c for c in comps if c in component_cols]
    for grp, comps in GROUPS.items()
}

comp_names   = list(component_cols.keys())
n_components = len(comp_names)

# Value matrix: shape (n_scenarios, n_components)
values = df.iloc[ROW_ORDER][list(component_cols.values())].values.astype(float)

# ── Color per component ────────────────────────────────────────────────────────
component_colors = {
    "PV":              "peru",
    "Wind":            "skyblue",
    "Electrolyser":    "tab:blue",
    "Fuel cell":       "gold",
    "Heat pump":       "tab:red",
    "Battery":         "purple",
    "Salt cavern":     "brown",
    "Thermal storage": "y",
}

# Colors per scenario (used only in Figure 1)
scenario_colors = ["black", "peru", "gold", "purple", "brown"]
scenario_colors = scenario_colors[:n_scenarios]
hatches         = [""] * n_scenarios

# One color per group (for composition / index figures)
group_colors = {
    "Supply":  "#55A868",
    "PtX":     "#4C72B0",
    "Storage": "#BD52DD",
}

# ── Helper: save figure ────────────────────────────────────────────────────────
fig_dir = (
    Path(__file__).parent.parent
    / "[03] Figures"
    / "Sensitivity_RemoveComponents"
)
fig_dir.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name: str):
    out = fig_dir / name
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: grouped bar chart — one figure per group, one cluster per scenario
# Each cluster has N bars (one per component), colored by component.
# ══════════════════════════════════════════════════════════════════════════════
def plot_group_clustered(group_name: str,
                         y_label: str,
                         filename: str):
    comps   = GROUPS[group_name]
    n_comps = len(comps)

    total_width = 0.7
    bar_w       = total_width / n_comps
    offsets     = np.linspace(
        -(total_width - bar_w) / 2,
         (total_width - bar_w) / 2,
        n_comps
    )

    x = np.arange(n_scenarios)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for k, comp in enumerate(comps):
        col_idx   = list(component_cols.keys()).index(comp)
        comp_vals = values[:, col_idx].copy()

        # Convert MWh → GWh for storage
        if group_name == "Storage":
            comp_vals = comp_vals / 1000.0

        ax.bar(
            x + offsets[k],
            comp_vals,
            width=bar_w * 0.92,
            color=component_colors[comp],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label=comp,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", direction="out")
    ax.legend(fontsize=8, loc="best")

    save_fig(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — All components (2×4 layout, colored by scenario)
# ══════════════════════════════════════════════════════════════════════════════
n_cols_all = 4
n_rows_all = int(np.ceil(n_components / n_cols_all))

fig1, axes1 = plt.subplots(n_rows_all, n_cols_all, figsize=(13, 5.5),
                            sharey=False, constrained_layout=True)
axes1 = axes1.flatten()
x = np.arange(n_scenarios)

for j, (comp, ax) in enumerate(zip(comp_names, axes1)):
    for i, (label, color, hatch) in enumerate(
            zip(scenario_labels, scenario_colors, hatches)):
        ax.bar(
            x[i], values[i, j],
            width=0.55,
            color=color,
            hatch=hatch,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_title(comp, fontsize=10, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", direction="out")

for ax in axes1[n_components:]:
    ax.set_visible(False)

for row_idx in range(n_rows_all):
    axes1[row_idx * n_cols_all].set_ylabel("Capacity\n[MW or MWh]", fontsize=9)

save_fig(fig1, "SENSITIVITY_component_capacities.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 2–4 — Per-group clustered bar charts (component colors)
# ══════════════════════════════════════════════════════════════════════════════
plot_group_clustered(
    "Supply",
    y_label="Capacity [MW]",
    filename="SENSITIVITY_supply_capacities.pdf",
)

plot_group_clustered(
    "PtX",
    y_label="Capacity [MW]",
    filename="SENSITIVITY_ptx_capacities.pdf",
)

plot_group_clustered(
    "Storage",
    y_label="Capacity [GWh]",
    filename="SENSITIVITY_storage_capacities.pdf",
)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 5 & 6 — Relative sizing across scenarios
# ══════════════════════════════════════════════════════════════════════════════

# ── Group sums ────────────────────────────────────────────────────────────────
def group_sum(group_name: str) -> np.ndarray:
    comps   = GROUPS[group_name]
    indices = [list(component_cols.keys()).index(c) for c in comps]
    return values[:, indices].sum(axis=1)

supply_sum  = group_sum("Supply")   # MW
ptx_sum     = group_sum("PtX")      # MW
storage_sum = group_sum("Storage")  # MWh

x_sc  = np.arange(n_scenarios)
xlbls = scenario_labels

# ── Figure 5 — Composition (100% stacked bars) ────────────────────────────────
fig5, (ax5l, ax5r) = plt.subplots(1, 2, figsize=(10, 4.5),
                                   constrained_layout=True)

# -- Left: Supply + PtX shares (MW) --
mw_total   = supply_sum + ptx_sum
supply_pct = np.where(mw_total > 0, supply_sum / mw_total * 100, 0)
ptx_pct    = np.where(mw_total > 0, ptx_sum    / mw_total * 100, 0)

bar_w = 0.5
ax5l.bar(x_sc, supply_pct, width=bar_w,
         color=group_colors["Supply"], label="Supply", alpha=0.85)
ax5l.bar(x_sc, ptx_pct, width=bar_w, bottom=supply_pct,
         color=group_colors["PtX"], label="PtX", alpha=0.85)

ax5l.set_ylim(0, 100)
ax5l.set_ylabel("Share of installed MW [%]", fontsize=9)
ax5l.set_xticks(x_sc)
ax5l.set_xticklabels(xlbls, fontsize=7.5, rotation=30, ha="right")
ax5l.set_axisbelow(True)
ax5l.legend(fontsize=8, loc="best")

# -- Right: Storage breakdown (MWh shares) --
storage_comps   = GROUPS["Storage"]
storage_indices = [list(component_cols.keys()).index(c) for c in storage_comps]
storage_vals    = values[:, storage_indices]
storage_totals  = storage_vals.sum(axis=1, keepdims=True)
storage_pct     = np.where(storage_totals > 0,
                            storage_vals / storage_totals * 100, 0)

# Use the component colors for the storage breakdown too
storage_bar_colors = [component_colors[c] for c in storage_comps]
bottoms = np.zeros(n_scenarios)
for k, (comp, col) in enumerate(zip(storage_comps, storage_bar_colors)):
    ax5r.bar(x_sc, storage_pct[:, k], width=bar_w,
             bottom=bottoms, color=col, label=comp, alpha=0.85)
    bottoms += storage_pct[:, k]

ax5r.set_ylim(0, 100)
ax5r.set_ylabel("Share of installed MWh [%]", fontsize=9)
ax5r.set_xticks(x_sc)
ax5r.set_xticklabels(xlbls, fontsize=7.5, rotation=30, ha="right")
ax5r.set_axisbelow(True)
ax5r.legend(fontsize=8, loc="best")

save_fig(fig5, "SENSITIVITY_relative_composition.pdf")


# ── Figure 6 — Index chart (base scenario = 100) ──────────────────────────────
BASE_IDX = 0

def to_index(arr: np.ndarray) -> np.ndarray:
    base_val = arr[BASE_IDX]
    return np.where(base_val > 0, arr / base_val , np.nan)

supply_idx  = to_index(supply_sum)
ptx_idx     = to_index(ptx_sum)
storage_idx = to_index(storage_sum)

# ── System cost for each scenario (index, base = 100) ─────────────────────────
system_cost_raw = df.iloc[ROW_ORDER]["financial.total_system_cost_eur_per_yr"].values.astype(float)
system_cost = system_cost_raw / system_cost_raw[BASE_IDX]

for i in scenario_labels:
    print(f"{i}: {system_cost_raw[scenario_labels.index(i)]:,.0f} EUR/yr (index: {system_cost[scenario_labels.index(i)]:.2f})")
    print("Scenario:", scenario_labels[scenario_labels.index(i)], "| System Cost Index:", system_cost[scenario_labels.index(i)])


fig6, ax6 = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax6_r = ax6.twinx()

bar_w6  = 0.22
offsets = [-bar_w6, 0, bar_w6]
groups6 = ["Supply", "PtX", "Storage"]
idx6    = [supply_idx, ptx_idx, storage_idx]

for offset, grp, idx_vals in zip(offsets, groups6, idx6):
    ax6.bar(x_sc + offset, idx_vals,
            width=bar_w6,
            color=group_colors[grp],
            alpha=0.85,
            label=f"{grp} Capacity")

#ax6.axhline(100, color="black", linewidth=0.8, linestyle="--")
ax6.set_ylabel("Capacity Index", fontsize=9)
ax6.set_xticks(x_sc)
ax6.set_xticklabels(xlbls, fontsize=8, rotation=30, ha="right")
ax6.set_axisbelow(True)

'''for offset, idx_vals in zip(offsets, idx6):
    for xi, val in zip(x_sc, idx_vals):
        if not np.isnan(val):
            ax6.text(xi + offset, val + 2.5, f"{val:.0f}",
                     ha="center", va="bottom", fontsize=6.5)'''

ax6_r.plot(x_sc, system_cost,
           color="black", linestyle="none",
           marker="o", markersize=5, zorder=5,
           label="Total System Cost")
ax6_r.plot(x_sc, system_cost,
           color="black", linestyle=":", linewidth=0.8, zorder=4)
ax6_r.set_ylabel("Total System Cost Index", fontsize=9)
ax6_r.tick_params(axis="y", direction="in")

# ── Combined legend ────────────────────────────────────────────────────────────
handles_l, labels_l = ax6.get_legend_handles_labels()
handles_r, labels_r = ax6_r.get_legend_handles_labels()
ax6.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8, loc="upper left")

# After all plotting, sync both y-axes
y_min = min(ax6.get_ylim()[0], ax6_r.get_ylim()[0])
y_max = max(ax6.get_ylim()[1], ax6_r.get_ylim()[1])
ax6.set_ylim(y_min, y_max)
ax6_r.set_ylim(y_min, y_max)

save_fig(fig6, "SENSITIVITY_relative_index.pdf")
plt.show()

