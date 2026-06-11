import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import funcs as func
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import matplotlib.lines as mlines

# ── Thesis styling ────────────────────────────────────────────────────────────
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

# ── Load data ─────────────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "Logbooks" / "Logbook_FINAL_MODEL_FIXEDCAPACITIES_FIXEDCURTAILMENT_NEW.csv"
df = pd.read_csv(csv_path)
df = df.sort_values("operational.curtailment_share")

x_col = "operational.curtailment_share"

# ── Storage capacities ────────────────────────────────────────────────────────
storage_cols = {
    "Battery":     "capacities.capacity_battery_energy_mwh",
    "Salt Cavern": "capacities.capacity_saltcavern_energy_mwh",
    "Thermal":     "capacities.capacity_thermalstorage_mwh",
}
df["total_storage_mwh"] = sum(df[c] for c in storage_cols.values())
storage_cols["Total"] = "total_storage_mwh"

storage_styles = {
    "Battery":     {"color": "purple", "marker": "o"},
    "Salt Cavern": {"color": "brown",  "marker": "s"},
    "Thermal":     {"color": "y",      "marker": "^"},
    "Total":       {"color": "black",  "marker": "D"},
}

# ── Utilisation columns ───────────────────────────────────────────────────────
utilisation_cols = {
    "Battery":     "operational.capacity_factor_battery_energy",
    "Salt Cavern": "operational.capacity_factor_saltcavern_energy",
    "Thermal":     "operational.capacity_factor_thermalstorage_energy",
}

# ── FIGURE 1: Storage installed vs curtailment share ─────────────────────────
fig1, ax1 = plt.subplots(figsize=(6, 4.5))

for label, col in storage_cols.items():
    s = storage_styles[label]
    ax1.plot(df[x_col], df[col],
             marker=s["marker"], color=s["color"],
             linestyle="--", markersize=4, label=label)

ax1.set_xlabel(r"Curtailment Share / [-]")
ax1.set_ylabel(r"Storage Installed / [MWh]")
ax1.legend(title="Storage type", fontsize=9, title_fontsize=10)

fig1.tight_layout()
output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / "CURTAILMENT_STORAGE_INSTALLED.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig1.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")

# ── FIGURE 2: Total system cost vs curtailment share ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4.5))

ax2.plot(df[x_col], df["financial.total_system_cost_eur_per_yr"],
         marker="o", color="black", linestyle="--", markersize=4)



ax2.set_xlabel(r"Curtailment Share / [-]")
ax2.set_ylabel(r"Total System Cost / [€ / year]")

fig2.tight_layout()
output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / "CURTAILMENT_SYSTEM_COST.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig2.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")
print("Curtailment share vs SYSTEM COSTS:")
print(df[["operational.curtailment_share", "financial.total_system_cost_eur_per_yr"]])


# ── FIGURE 2a: Total system cost vs curtailment share
# ── FIGURE 2a: Total system cost vs curtailment share
fig2a, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5))

x = df[x_col]
y = df["financial.total_system_cost_eur_per_yr"]
min_idx = y.idxmin()
min_x = df.loc[min_idx, x_col]
min_y = y.loc[min_idx]

# --- (a) Left: full picture ---
ax_full.plot(x, y/1e6, marker="o", color="black", linestyle="--", markersize=4)
ax_full.set_xlabel(r"Curtailment Share / [-]", fontsize=15)
ax_full.set_ylabel(r"Total System Cost / [M€ / year]", fontsize=15)
ax_full.tick_params(axis="both", labelsize=15)
ax_full.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
ax_full.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax_full.text(-0.1, 1.05, "(a)", transform=ax_full.transAxes,
             fontsize=15, fontweight="bold", va="top", ha="center")

# --- (b) Right: zoomed into the flat region ---
zoom_mask = x >= 0.55
ax_zoom.plot(x[zoom_mask], y[zoom_mask]/1e6, marker="o", color="black", linestyle="--", markersize=4)
ax_zoom.set_xlabel(r"Curtailment Share / [-]", fontsize=15)
ax_zoom.set_ylabel(r"Total System Cost / [M€ / year]", fontsize=15)
ax_zoom.tick_params(axis="both", labelsize=15)
ax_zoom.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
ax_zoom.legend(fontsize=16)
ax_zoom.text(-0.1, 1.05, "(b)", transform=ax_zoom.transAxes,
             fontsize=15, fontweight="bold", va="top", ha="center")

fig2a.tight_layout()
output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / "CURTAILMENT_SYSTEM_COST_DETAIL.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig2a.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")







# ── FIGURE 3: Storage utilisation vs curtailment share ───────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 4.5))

for label, col in utilisation_cols.items():
    s = storage_styles[label]
    ax3.plot(df[x_col], df[col],
             marker=s["marker"], color=s["color"],
             linestyle="--", markersize=4, label=label)

ax3.set_xlabel(r"Curtailment Share / [-]")
ax3.set_ylabel(r"Storage Utilisation / [-]")
ax3.legend(title="Storage type", fontsize=9, title_fontsize=10)

fig3.tight_layout()
output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / "CURTAILMENT_STORAGE_UTILISATION.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig3.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")



# ── FIGURE 1+3 combined: Storage installed (stacked fill) + utilisation lines ─
fig_combo, ax_left = plt.subplots(figsize=(8, 5))
ax_right = ax_left.twinx()

x = df[x_col]

# Stacked fill order: Salt Cavern (bottom), Battery (middle), Thermal (top)
stack_order = ["Salt Cavern", "Battery", "Thermal Storage"]
stack_cols  = {
    "Salt Cavern":     "capacities.capacity_saltcavern_energy_mwh",
    "Battery":         "capacities.capacity_battery_energy_mwh",
    "Thermal Storage": "capacities.capacity_thermalstorage_mwh",
}
stack_colors = {
    "Salt Cavern":     "brown",
    "Battery":         "purple",
    "Thermal Storage": "y",
}
component_markers = {
    "Salt Cavern":     "s",
    "Battery":         "o",
    "Thermal Storage": "^",
}

bottom = pd.Series(0.0, index=df.index)
for label in stack_order:
    col  = stack_cols[label]
    vals = df[col] / 1e3  # MWh → GWh
    top  = bottom + vals
    ax_left.fill_between(x, bottom, top,
                         color=stack_colors[label],
                         alpha=0.4, linewidth=0)
    bottom = top

ax_left.set_xlabel(r"Curtailment Share / [-]")
ax_left.set_ylabel(r"Storage Capacity / [GWh]")

# Utilisation lines on the right axis
for label, col in utilisation_cols.items():
    s = storage_styles[label]
    ax_right.plot(x, df[col],
                  marker=s["marker"], color=s["color"],
                  linestyle="--", linewidth=1.5, markersize=4)

ax_right.set_ylabel(r"Average SOC / [-]")
ax_right.set_ylim(0, 1)
ax_left.set_ylim(bottom=0)

# ── Combined legend: patch + line per component ───────────────────────────────
combined_handles = []
for label in stack_order:
    color = stack_colors[label]
    patch = mpatches.Patch(facecolor=color, alpha=0.4, linewidth=0)
    line  = mlines.Line2D([], [], color=color, marker=component_markers[label],
                          linestyle="--", linewidth=1.5, markersize=4)
    combined_handles.append((patch, line))

ax_left.legend(
    combined_handles,
    stack_order,
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
    fontsize=9,
    loc="upper right",
    frameon=True,
)

fig_combo.tight_layout()

output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / "CURTAILMENT_STORAGE_COMBINED.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig_combo.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")


'''fig_combo, ax_left = plt.subplots(figsize=(8, 5))
ax_right = ax_left.twinx()

x = df[x_col]

# Stacked fill order: Salt Cavern (bottom), Battery (middle), Thermal (top)
stack_order = ["Salt Cavern", "Battery", "Thermal Storage"]
stack_cols  = {
    "Salt Cavern":     "capacities.capacity_saltcavern_energy_mwh",
    "Battery":         "capacities.capacity_battery_energy_mwh",
    "Thermal Storage": "capacities.capacity_thermalstorage_mwh",
}
stack_colors = {
    "Salt Cavern":     "brown",
    "Battery":         "purple",
    "Thermal Storage": "y",
}

# Maps display names (used in stack_order) → keys in storage_styles
style_keys = {
    "Salt Cavern":     "Salt Cavern",
    "Battery":         "Battery",
    "Thermal Storage": "Thermal",
}

bottom = pd.Series(0.0, index=df.index)
for label in stack_order:
    col  = stack_cols[label]
    vals = df[col] / 1e3  # MWh → GWh
    top  = bottom + vals
    ax_left.fill_between(x, bottom, top,
                         color=stack_colors[label],
                         alpha=0.4, linewidth=0)
    bottom = top

ax_left.set_xlabel(r"Curtailment Share / [-]")
ax_left.set_ylabel(r"Storage Capacity / [GWh]")

# Utilisation lines on the right axis
for label, col in utilisation_cols.items():
    s = storage_styles[label]
    ax_right.plot(x, df[col],
                  marker=s["marker"], color=s["color"],
                  linestyle="--", linewidth=1.5, markersize=4)

ax_right.set_ylabel(r"Storage Utilisation / [-]")
ax_right.set_ylim(0, 1)
ax_left.set_ylim(bottom=0)

# ── Single condensed legend: patch + dashed line per row ─────────────────────
combined_handles = []
for label in stack_order:
    color  = stack_colors[label]
    marker = storage_styles[style_keys[label]]["marker"]
    patch  = mpatches.Patch(facecolor=color, alpha=0.4, linewidth=0)
    line   = mlines.Line2D([], [], color=color, marker=marker,
                           linestyle="--", linewidth=1.5, markersize=4)
    combined_handles.append((patch, line))

ax_left.legend(
    combined_handles,
    stack_order,
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
    title="fill = capacity [GWh]  |  line = utilisation [-]",
    title_fontsize=8,
    fontsize=9,
    loc="upper right",
    frameon=False,
)

fig_combo.tight_layout()

output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / "CURTAILMENT_STORAGE_COMBINED.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig_combo.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")'''


# ── FIGURE 4: Storage costs by component vs curtailment share ─────────────────

# Derived cost columns
df["cost_battery_eur"] = (
    df["capex_undiscounted.battery_total_eur"] +
    df["opex_fixed_eur_per_yr.battery"] +
    df["opex_variable_eur_per_yr.battery_charge"] +
    df["opex_variable_eur_per_yr.battery_discharge"]
)

df["cost_saltcavern_eur"] = (
    df["capex_undiscounted.saltcavern_geological_eur"] +
    df["capex_undiscounted.saltcavern_cushiongas_eur"] +
    df["capex_undiscounted.saltcavern_compressor_eur"] +
    df["opex_fixed_eur_per_yr.saltcavern"] +
    df["opex_variable_eur_per_yr.saltcavern_compress"]
)

df["cost_thermal_eur"] = (
    df["capex_undiscounted.thermalstorage_eur"] +
    df["opex_fixed_eur_per_yr.thermalstorage"]
)

cost_cols = {
    "Battery":     "cost_battery_eur",
    "Salt Cavern": "cost_saltcavern_eur",
    "Thermal":     "cost_thermal_eur",
}

fig4, ax4 = plt.subplots(figsize=(6, 4.5))

for label, col in cost_cols.items():
    s = storage_styles[label]
    ax4.plot(df[x_col], df[col],
             marker=s["marker"], color=s["color"],
             linestyle="--", markersize=4, label=label)

ax4.set_xlabel(r"Curtailment Share / [-]")
ax4.set_ylabel(r"Storage Cost / [€]")
ax4.legend(title="Storage type", fontsize=9, title_fontsize=10)

fig4.tight_layout()
output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / "CURTAILMENT_STORAGE_COSTS.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig4.savefig(output, bbox_inches="tight")
print(f"Saved: {output}")
plt.show()









# ── FIGURE 5 & 6: Flexibility metrics vs curtailment share ───────────────────

r_pre = "flexibility_radar."

storage_metrics_radar = {
    "Responsiveness":    ("battery.responsiveness",   "salt_cavern.responsiveness",   "thermal_storage.responsiveness"),
    "Utilisation":       ("battery.utilisation",      "salt_cavern.utilisation",      "thermal_storage.utilisation"),
    "Upward flex":       ("battery.upward_flex",      "salt_cavern.upward_flex",      "thermal_storage.upward_flex"),
    "Downward flex":     ("battery.downward_flex",    "salt_cavern.downward_flex",    "thermal_storage.downward_flex"),
    "SSR":               ("battery.ssr",              "salt_cavern.ssr",              "thermal_storage.ssr"),
    "Load shift index":  ("battery.load_shift_index", "salt_cavern.load_shift_index", "thermal_storage.load_shift_index"),
}

ptx_metrics_radar = {
    "Responsiveness":    ("electrolyser.responsiveness",  "heat_pump.responsiveness",  "fuelcell.responsiveness"),
    "Capacity factor":   ("electrolyser.capacity_factor", "heat_pump.capacity_factor", "fuelcell.capacity_factor"),
    "Upward flex":       ("electrolyser.upward_flex",     "heat_pump.upward_flex",     "fuelcell.upward_flex"),
    "Downward flex":     ("electrolyser.downward_flex",   "heat_pump.downward_flex",   "fuelcell.downward_flex"),
    "SSR":               ("electrolyser.ssr",             "heat_pump.ssr",             "fuelcell.ssr"),
    "SCR":               ("electrolyser.scr",             "heat_pump.scr",             "fuelcell.scr"),
}

storage_components_radar = {
    "Battery":         {"color": "purple", "marker": "o"},
    "Salt Cavern":     {"color": "brown",  "marker": "s"},
    "Thermal Storage": {"color": "y",      "marker": "^"},
}

ptx_components_radar = {
    "Electrolyser": {"color": "tab:blue",  "marker": "o"},
    "Heat Pump":    {"color": "tab:red",   "marker": "s"},
    "Fuel Cell":    {"color": "gold",      "marker": "^"},
}

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

def plot_flex_evolution(metrics_dict, components_dict, df, x_col, fig_name):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    comp_labels = list(components_dict.keys())

    for idx, (ax, (metric_label, cols)) in enumerate(
            zip(axes.flat, metrics_dict.items())):

        for col_suffix, label in zip(cols, comp_labels):
            col = r_pre + col_suffix
            if col in df.columns:
                s = components_dict[label]
                ax.plot(df[x_col], df[col],
                        marker=s["marker"], color=s["color"],
                        linestyle="--", markersize=4, label=label)

        ax.set_ylim(0, 1)
        ax.set_xlabel(r"Curtailment Share / [-]")
        ax.set_ylabel(metric_label + r" / [-]")
        ax.text(0.02, 0.02, panel_labels[idx], transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")

    # single shared legend on last axis
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, title="Component",
                         fontsize=9, title_fontsize=10)

    fig.tight_layout()

    output = Path(__file__).parent.parent / "[03] Figures" / "Curtailment" / f"{fig_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.show()


plot_flex_evolution(storage_metrics_radar, storage_components_radar,
                    df, x_col, "CURTAILMENT_FLEX_STORAGE_EVOLUTION")

plot_flex_evolution(ptx_metrics_radar, ptx_components_radar,
                    df, x_col, "CURTAILMENT_FLEX_PTX_EVOLUTION")