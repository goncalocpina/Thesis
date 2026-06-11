"""
Carbon Break-Even Analysis
==========================
Evaluates the competitiveness of a renewable-based energy system against
conventional production pathways, by computing the CO2 price required to
close the cost gap for each energy carrier and at the system level.

Methodology
-----------
Per carrier:
    P_CO2 = (P_sell - P_market) / e_avoided                          [€/tCO2]
    ΔP    = P_CO2 - P_ETS                                            [€/tCO2]

System-level:
    P_CO2,sys = Σ_i [D_i (P_sell,i - P_market,i)] / Σ_i [D_i e_avoided,i]  [€/tCO2]
    ΔP_sys    = P_CO2,sys - P_ETS                                    [€/tCO2]

Units
-----
All carriers are expressed on a MWh basis throughout calculations and plots.
H2 values are additionally shown in €/kg (in parentheses) in the print output.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass
from pathlib import Path
from Parameters import parameters as par
import matplotlib as mpl

# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent

demand_file = DATA_DIR.parent / "[02] Data" / "Demand" / "Demand_Profiles.csv"

df_dem = pd.read_csv(demand_file, sep=";")
df_dem["Datetime (UTC)"] = pd.to_datetime(
    df_dem["Datetime (UTC)"], format="%d/%m/%Y %H:%M"
)
df_dem = df_dem[[
    "Datetime (UTC)",
    "Rotterdam_total_gas_demand [MW]",
    "Rotterdam_electricity_load [MW]",
    "Rotterdam_total_heat_demand [MW]",
]].copy()
for col in df_dem.columns[1:]:
    df_dem[col] = pd.to_numeric(df_dem[col], errors="coerce")
df_dem = df_dem.dropna().sort_values("Datetime (UTC)").reset_index(drop=True)

# Build hourly timeseries [MWh/h → summed to MWh/yr]
timeseries = pd.DataFrame({
    "H2_Demand":          df_dem["Rotterdam_total_gas_demand [MW]"].values,
    "Electricity_Demand": df_dem["Rotterdam_electricity_load [MW]"].values,
    "Heat_Demand":        df_dem["Rotterdam_total_heat_demand [MW]"].values,
}, index=pd.DatetimeIndex(df_dem["Datetime (UTC)"]).round("h"))
timeseries.index.name = "time"
timeseries = timeseries.asfreq("h")

print(f"Demand profiles loaded: {len(timeseries)} rows\n")
for col in timeseries.columns:
    print(f"  {col}: max = {timeseries[col].max():.1f} MW | "
          f"annual = {timeseries[col].sum():,.0f} MWh")


# All demands expressed in MWh/yr
demand_elec_MWh = timeseries["Electricity_Demand"].sum()   # [MWh/yr]
demand_h2_MWh   = timeseries["H2_Demand"].sum()            # [MWh/yr]
demand_heat_MWh = timeseries["Heat_Demand"].sum()          # [MWh/yr]

# =============================================================================
# SECTION 2 — INPUT PARAMETERS
# =============================================================================

# ── Selling prices (MWh basis throughout) ────────────────────────────────────
P_sell_elec = 752.51    # [€/MWh]
P_sell_h2   = 321.44    # [€/MWh]  — energy basis (primary)
P_sell_heat = 215.05    # [€/MWh]

# ── Conventional market reference prices (MWh basis) ─────────────────────────
P_market_elec = 247                                # [€/MWh]  — day-ahead average
P_market_h2   = 7.547453 / par.H2_CALORIFIC_VALUE_LHV    # [€/MWh]  — SMR hydrogen (converted from €/kg)
P_market_heat = 184.13                                # [€/MWh]  — natural gas boiler (gas price / boiler efficiency)

# ── CO2 emission intensities of conventional production (MWh basis) ───────────
# Represent avoided emissions per unit of energy when using the renewable
# system instead of the conventional pathway.
e_elec = 0.300          # [tCO2/MWh]  — NL grid emission factor
e_h2   = e_elec /par.ELECTROLYSER_EFFICIENCY      # [tCO2/MWh] — SMR without CCS (converted from kg_CO2/kg_H2)
e_heat = 0.2034         # [tCO2/MWh]  — natural gas emission factor

# ── EU ETS carbon price reference ────────────────────────────────────────────
P_ETS = 75.0            # [€/tCO2]

# =============================================================================
# SECTION 3 — CARRIER DEFINITIONS
# =============================================================================

@dataclass
class Carrier:
    name:   str
    demand: float   # D_i         [MWh/yr]
    P_sell: float   # P_sell,i    [€/MWh]
    P_mkt:  float   # P_market,i  [€/MWh]
    e_av:   float   # e_avoided,i [tCO2/MWh]

carriers = [
    Carrier("Electricity", demand_elec_MWh, P_sell_elec, P_market_elec, e_elec),
    Carrier("H$_2$",    demand_h2_MWh,   P_sell_h2,   P_market_h2,   e_h2),
    Carrier("Heat",        demand_heat_MWh, P_sell_heat, P_market_heat, e_heat),
]

# =============================================================================
# SECTION 4 — PER-CARRIER CALCULATIONS  (all on MWh basis)
# =============================================================================
#
#   price_gap = P_sell - P_market                          [€/MWh]
#   P_CO2     = price_gap / e_avoided                      [€/tCO2]
#   ΔP        = P_CO2 - P_ETS                              [€/tCO2]
#
#   cost_gap_total    = D * price_gap                      [€/yr]
#   co2_avoided_total = D * e_avoided                      [tCO2/yr]

results = []
for c in carriers:

    # Price gap between selling price and conventional market price
    price_gap = c.P_sell - c.P_mkt                        # [€/MWh]

    # Break-even CO2 price: CO2 value needed to close the per-unit price gap
    # P_CO2 = (P_sell - P_market) / e_avoided              [€/tCO2]
    P_CO2 = price_gap / c.e_av                             # [€/tCO2]

    # CO2 cost gap: additional carbon price needed beyond the current ETS level
    # ΔP = P_CO2 - P_ETS                                   [€/tCO2]
    delta_P = P_CO2 - P_ETS                                # [€/tCO2]

    # Volumetric quantities — used in system-level aggregate (Section 5)
    cost_gap_total    = c.demand * price_gap               # [€/yr]
    co2_avoided_total = c.demand * c.e_av                  # [tCO2/yr]

    results.append({
        "carrier":           c.name,
        "demand":            c.demand,            # [MWh/yr]
        "P_sell":            c.P_sell,            # [€/MWh]
        "P_mkt":             c.P_mkt,             # [€/MWh]
        "e_av":              c.e_av,              # [tCO2/MWh]
        "price_gap":         price_gap,           # [€/MWh]
        "P_CO2":             P_CO2,               # [€/tCO2]
        "delta_P":           delta_P,             # [€/tCO2]
        "cost_gap_total":    cost_gap_total,      # [€/yr]
        "co2_avoided_total": co2_avoided_total,   # [tCO2/yr]
    })

# =============================================================================
# SECTION 5 — SYSTEM-LEVEL AGGREGATE
# =============================================================================
#
#   P_CO2,sys = Σ_i [D_i (P_sell,i - P_market,i)] / Σ_i [D_i e_avoided,i]
#             = Total cost gap / Total CO2 avoided         [€/tCO2]
#
#   ΔP_sys = P_CO2,sys - P_ETS                            [€/tCO2]

total_demand      = sum(r["demand"]            for r in results)   # [MWh/yr]
total_cost_gap    = sum(r["cost_gap_total"]    for r in results)   # [€/yr]
total_co2_avoided = sum(r["co2_avoided_total"] for r in results)   # [tCO2/yr]

# System-level break-even CO2 price (demand-weighted average across carriers)
P_CO2_sys   = total_cost_gap / total_co2_avoided                   # [€/tCO2]

# System-level CO2 cost gap
delta_P_sys = P_CO2_sys - P_ETS                                    # [€/tCO2]

# Demand-weighted system averages for €/MWh representation of panel (b)
#   P_sell_sys(P_CO2) = Σ(D_i · P_sell,i) / Σ D_i                 [€/MWh]  — flat
#   P_conv_sys(P_CO2) = Σ(D_i · P_mkt,i)  / Σ D_i
#                     + [Σ(D_i · e_i) / Σ D_i] · P_CO2            [€/MWh]  — rising
# The intersection of these two lines occurs at exactly P_CO2,sys, consistent
# with the volumetric formulation in Section 5.
P_sell_sys_avg = sum(r["demand"] * r["P_sell"] for r in results) / total_demand  # [€/MWh]
P_mkt_sys_avg  = sum(r["demand"] * r["P_mkt"]  for r in results) / total_demand  # [€/MWh]
e_sys_avg      = sum(r["demand"] * r["e_av"]   for r in results) / total_demand  # [tCO2/MWh]

# =============================================================================
# SECTION 6 — PRINT RESULTS
# =============================================================================

SEP = "─" * 68

def fmt_co2(val):
    """Format a CO2 break-even price with ETS comparison."""
    if val <= 0:
        return "Already competitive"
    flag = "✓ below ETS" if val <= P_ETS else f"✗ ETS gap: +€{val - P_ETS:.0f}/t\u2009CO\u2082"
    return f"€{val:.1f}/t\u2009CO\u2082  ({flag})"

def h2_kg(val_per_MWh):
    """Convert a €/MWh H2 value to €/kg for display in parentheses."""
    return val_per_MWh * par.H2_CALORIFIC_VALUE_LHV

print(f"\n{SEP}")
print("  PER-CARRIER RESULTS")
print(SEP)
for r in results:
    is_h2 = r["carrier"] == "H$_2$"
    print(f"\n  {r['carrier']}")

    if is_h2:
        print(f"    P_sell            : €{r['P_sell']:.2f}/MWh  (€{h2_kg(r['P_sell']):.2f}/kg)")
        print(f"    P_market          : €{r['P_mkt']:.2f}/MWh  (€{h2_kg(r['P_mkt']):.2f}/kg)")
        print(f"    Price gap         : €{r['price_gap']:+.2f}/MWh  (€{h2_kg(r['price_gap']):+.2f}/kg)")
        print(f"    e_avoided         : {r['e_av']:.4f} t\u2009CO\u2082/MWh  ({h2_kg(r['e_av']):.4f} t\u2009CO\u2082/kg)")
    else:
        print(f"    P_sell            : €{r['P_sell']:.2f}/MWh")
        print(f"    P_market          : €{r['P_mkt']:.2f}/MWh")
        print(f"    Price gap         : €{r['price_gap']:+.2f}/MWh")
        print(f"    e_avoided         : {r['e_av']:.4f} t\u2009CO\u2082/MWh")

    print(f"    Break-even P_CO2  : {fmt_co2(r['P_CO2'])}")
    print(f"    ΔP (gap to ETS)   : €{r['delta_P']:+.1f}/t\u2009CO\u2082")
    print(f"    ── Volumetric ──")
    print(f"    Annual demand     : {r['demand']:>15,.0f} MWh/yr")
    print(f"    CO\u2082 avoided       : {r['co2_avoided_total']:>15,.1f} t\u2009CO\u2082/yr")
    print(f"    Total cost gap    : €{r['cost_gap_total']:>14,.0f}/yr")

print(f"\n{SEP}")
print("  SYSTEM-LEVEL AGGREGATE")
print(SEP)
print(f"    Total CO\u2082 avoided  : {total_co2_avoided:>15,.1f} t\u2009CO\u2082/yr")
print(f"    Total cost gap     : €{total_cost_gap:>14,.0f}/yr")
print(f"    System P_CO2,sys   : {fmt_co2(P_CO2_sys)}")
print(f"    ΔP_sys (gap to ETS): €{delta_P_sys:+.1f}/t\u2009CO\u2082")
print(f"    EU ETS reference   : €{P_ETS:.0f}/t\u2009CO\u2082")
print(SEP)

# =============================================================================
# SECTION 7 — PLOTS
# =============================================================================

COLORS    = ["tab:olive", "tab:green", "tab:orange"]
all_be    = [r["P_CO2"] for r in results if r["P_CO2"] > 0]
MAX_X     = np.ceil(max(*all_be, P_CO2_sys, P_ETS) * 1.5 / 50) * 50
co2_range = np.linspace(0, MAX_X, 400)

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# ── Left panel: per-carrier break-even ───────────────────────────────────────
# Conventional cost rises linearly with CO2 price; selling price is flat.
# All carriers on MWh basis. Intersection = per-carrier break-even P_CO2.
for r, col in zip(results, COLORS):
    conv_cost = r["P_mkt"] + r["e_av"] * co2_range    # P_market + e * P_CO2  [€/MWh]
    ax1.plot(co2_range, conv_cost, color=col, linewidth=1.2,
             label=f"{r['carrier']} (conventional)")
    ax1.axhline(r["P_sell"], color=col, linewidth=1.2, linestyle="--",
                label=f"{r['carrier']} (system price)")
    if 0 < r["P_CO2"] <= MAX_X:
        # Intersection point: both lines meet at (P_CO2, P_sell)
        ax1.scatter(r["P_CO2"], r["P_sell"], color=col, s=50, zorder=5,
                    edgecolors="white", linewidths=0.8)

ax1.axvline(P_ETS, color="#888", linewidth=0.8, linestyle="--",
            label=f"EU ETS")
ax1.set_xlabel(r"CO$_2$ price / [€/ton$_{\text{CO}_2}$]")
ax1.set_ylabel("Cost / [€/MWh]")
ax1.legend(fontsize=8, loc="upper center")
ax1.set_xlim(0, MAX_X)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}"))
ax1.tick_params(axis="both", direction="in")
ax1.set_axisbelow(True)

# ── Right panel: system-level on €/MWh basis ─────────────────────────────────
# Demand-weighted average system selling price (flat) vs rising demand-weighted
# conventional cost. Both expressed in €/MWh so the y-axis is identical to ax1.
# Intersection at exactly P_CO2,sys (consistent with Section 5 aggregate).
conv_cost_sys = P_mkt_sys_avg + e_sys_avg * co2_range   # [€/MWh]

ax2.plot(co2_range, conv_cost_sys, color="slateblue", linewidth=1.2,
         label="System (conventional, weighted avg)")
ax2.axhline(P_sell_sys_avg, color="slateblue", linewidth=1.2, linestyle="--",
            label=f"System (selling price, weighted avg)")
ax2.axvline(P_ETS, color="#888", linewidth=0.8, linestyle="--",
            label=f"EU ETS (€{P_ETS:.0f}/t\u2009CO\u2082)")
if 0 < P_CO2_sys <= MAX_X:
    # Intersection point: both system lines meet at (P_CO2_sys, P_sell_sys_avg)
    ax2.scatter(P_CO2_sys, P_sell_sys_avg, color="slateblue", s=50, zorder=5,
                edgecolors="white", linewidths=0.8)

ax2.set_xlabel(r"CO$_2$ price / [€/ton$_{\text{CO}_2}$]")
ax2.set_ylabel("Cost / [€/MWh]")
ax2.legend(fontsize=8, loc="best")
ax2.set_xlim(0, MAX_X)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}"))
ax2.tick_params(axis="both", direction="in")
ax2.set_axisbelow(True)

ax1.text(-0.02, 1.02, "(a)", transform=ax1.transAxes,
         fontsize=11, fontfamily="serif", fontweight="bold", va="bottom", ha="right")
ax2.text(-0.02, 1.02, "(b)", transform=ax2.transAxes,
         fontsize=11, fontfamily="serif", fontweight="bold", va="bottom", ha="right")

fig.tight_layout()

output = Path(__file__).parent.parent / "[03] Figures" / "CarbonBreakEven" / "CarbonBreakEven_old.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
print(f"\nFigure saved → {output}")
plt.show()

# =============================================================================
# SECTION 8 — COMBINED SINGLE-PANEL PLOT  (CarbonBreakEven1)
# =============================================================================
#
#   Single axis, all curves in €/MWh.
#   Per-carrier lines (solid = conventional rising, dashed = system selling price).
#   System aggregate lines in slateblue, same convention.
#   Legend placed outside the axes, to the right. No (a)/(b) labels.

fig1, ax = plt.subplots(figsize=(10, 5))

# ── Per-carrier lines ─────────────────────────────────────────────────────────
for r, col in zip(results, COLORS):
    conv_cost = r["P_mkt"] + r["e_av"] * co2_range    # [€/MWh]
    ax.plot(co2_range, conv_cost, color=col, linewidth=1.2, zorder=9,
            label=f"Conventional {r['carrier']} Cost")
    ax.axhline(r["P_sell"], color=col, linewidth=1.2, linestyle="--",
               label=f"System {r['carrier']} Cost", zorder=7)
    if 0 < r["P_CO2"] <= MAX_X:
        # Intersection point: both lines meet at (P_CO2, P_sell)
        ax.scatter(r["P_CO2"], r["P_sell"], color=col, s=50, zorder=10,
                   edgecolors="white", linewidths=0.8)

conv_cost_sys = P_mkt_sys_avg + e_sys_avg * co2_range   # [€/MWh]
ax.plot(co2_range, conv_cost_sys, color="slateblue", linewidth=1.2, zorder=8,
        label="Conventional Energy Cost\n (Demand-Weighted Avg.)")
ax.axhline(P_sell_sys_avg, color="slateblue", linewidth=1.2, linestyle="--",
           label="System Energy Cost\n (Demand-Weighted Avg.)")
if 0 < P_CO2_sys <= MAX_X:
    # Intersection point: both system lines meet at (P_CO2_sys, P_sell_sys_avg)
    ax.scatter(P_CO2_sys, P_sell_sys_avg, color="slateblue", s=50, zorder=4,
               edgecolors="white", linewidths=0.8)

# ── EU ETS reference ──────────────────────────────────────────────────────────
ax.axvline(P_ETS, color="black", linewidth=0.8, linestyle="--", label=f"EU ETS", zorder = 1)

# ── Axis formatting ───────────────────────────────────────────────────────────
ax.set_xlabel(r"CO$_2$ price / [€/ton$_{\text{CO}_2}$]", fontsize=12)
ax.set_ylabel("Cost / [€/MWh]", fontsize=12)
ax.set_xlim(0, MAX_X)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}"))
ax.tick_params(axis="both", direction="in", labelsize=12)
ax.set_axisbelow(True)

# ── Legend outside to the right ───────────────────────────────────────────────
ax.legend(
    fontsize=12,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
)

fig1.tight_layout()

output1 = Path(__file__).parent.parent / "[03] Figures" / "CarbonBreakEven" / "CarbonBreakEven.pdf"
output1.parent.mkdir(parents=True, exist_ok=True)
fig1.savefig(output1, bbox_inches="tight")
print(f"Combined figure saved → {output1}")
plt.show()