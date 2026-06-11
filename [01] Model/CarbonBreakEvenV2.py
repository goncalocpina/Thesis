"""
Carbon Break-Even Analysis
==========================
Evaluates the competitiveness of a renewable-based energy system against
conventional production pathways.

Structure
---------
- Electricity and H2: per-carrier carbon break-even analysis
      P_CO2,i  = (P_sell,i - P_market,i) / e_avoided,i       [€/tCO2]
      ΔP_i     = P_CO2,i - P_ETS                              [€/tCO2]

- Heat: simple cost comparison only (no per-carrier carbon break-even),
  as district heat has no liquid traded market with a well-defined emission
  factor. A conventional reference (natural gas boiler) is used solely for
  the system-level aggregate, with the assumption stated explicitly.

- System-level (all three carriers):
      P_CO2,sys = Σ_i [D_i (P_sell,i - P_market,i)]
                / Σ_i [D_i e_avoided,i]                       [€/tCO2]
      ΔP_sys    = P_CO2,sys - P_ETS                           [€/tCO2]
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

timeseries = pd.DataFrame({
    "H2_Demand":          df_dem["Rotterdam_total_gas_demand [MW]"].values,
    "Electricity_Demand": df_dem["Rotterdam_electricity_load [MW]"].values,
    "Heat_Demand":        df_dem["Rotterdam_total_heat_demand [MW]"].values,
}, index=pd.DatetimeIndex(df_dem["Datetime (UTC)"]).round("h"))
timeseries.index.name = "time"
timeseries = timeseries.asfreq("h")

print(f"Demand profiles loaded: {len(timeseries)} rows")

# ── H2 unit conversion: MWh → kg ─────────────────────────────────────────────
# H2 lower heating value (LHV) = 33.33 kWh/kg → 1 MWh = 1000/33.33 kg
H2_LHV_kWh_per_kg = 33.33                              # [kWh/kg]
MWh_to_kg         = 1000.0 / H2_LHV_kWh_per_kg         # [kg/MWh]

demand_elec_MWh = timeseries["Electricity_Demand"].sum()          # [MWh/yr]
demand_h2_kg    = timeseries["H2_Demand"].sum() * MWh_to_kg       # [kg/yr]
demand_heat_MWh = timeseries["Heat_Demand"].sum()                 # [MWh/yr]

# =============================================================================
# SECTION 2 — INPUT PARAMETERS
# =============================================================================

# ── Selling prices (from model output) ───────────────────────────────────────
P_sell_elec = 752.51    # [€/MWh]
P_sell_h2   = 10.71     # [€/kg]
P_sell_heat = 215.05    # [€/MWh]

# ── Conventional market reference prices ─────────────────────────────────────
P_market_elec = 62.0    # [€/MWh]  — Iberian OMIE day-ahead average
P_market_h2   = 5.2     # [€/kg]   — SMR hydrogen reference
P_market_heat = 38.0    # [€/MWh]  — natural gas boiler equivalent (assumption;
                         #             used only for system-level aggregate)

# ── CO2 emission intensities of conventional production ───────────────────────
# Represent the avoided emissions per unit of carrier when using the
# renewable-based system instead of the conventional pathway.
e_elec = 0.18           # [tCO2/MWh]  — Portuguese grid emission factor
e_h2   = 0.010          # [tCO2/kg]   — SMR without CCS
e_heat = 0.20           # [tCO2/MWh]  — natural gas boiler (assumption;
                         #               used only for system-level aggregate)

# ── EU ETS carbon price reference ────────────────────────────────────────────
P_ETS = 75.0            # [€/tCO2]

# =============================================================================
# SECTION 3 — CARRIER DEFINITIONS
# =============================================================================

@dataclass
class Carrier:
    name:            str
    unit:            str
    demand:          float   # D_i        [unit/yr]
    P_sell:          float   # P_sell,i   [€/unit]
    P_mkt:           float   # P_market,i [€/unit]
    e_av:            float   # e_avoided,i [tCO2/unit]
    carbon_analysis: bool    # True → per-carrier P_CO2 is reported
                             # False → cost comparison only (heat)

carriers = [
    Carrier("Electricity", "MWh", demand_elec_MWh, P_sell_elec, P_market_elec, e_elec, carbon_analysis=True),
    Carrier("Hydrogen",    "kg",  demand_h2_kg,    P_sell_h2,   P_market_h2,   e_h2,   carbon_analysis=True),
    Carrier("Heat",        "MWh", demand_heat_MWh, P_sell_heat, P_market_heat, e_heat, carbon_analysis=False),
]

# =============================================================================
# SECTION 4 — PER-CARRIER CALCULATIONS
# =============================================================================
#
# For carriers with carbon_analysis = True (electricity, H2):
#
#   price_gap = P_sell,i - P_market,i                     [€/unit]
#   P_CO2,i   = price_gap / e_avoided,i                   [€/tCO2]
#   ΔP_i      = P_CO2,i - P_ETS                           [€/tCO2]
#
# For heat (carbon_analysis = False):
#   price_gap reported only; P_CO2 and ΔP are not computed.
#
# For all carriers (needed for system-level aggregate in Section 5):
#   cost_gap_total,i    = D_i * price_gap,i                [€/yr]
#   co2_avoided_total,i = D_i * e_avoided,i                [tCO2/yr]

results = []
for c in carriers:

    # Price gap between selling price and conventional market price
    price_gap = c.P_sell - c.P_mkt                        # [€/unit]

    if c.carbon_analysis:
        # Break-even CO2 price: CO2 value needed to close the per-unit price gap
        # P_CO2 = (P_sell - P_market) / e_avoided          [€/tCO2]
        P_CO2 = price_gap / c.e_av                         # [€/tCO2]

        # CO2 cost gap: additional carbon price needed beyond current ETS level
        # ΔP = P_CO2 - P_ETS                               [€/tCO2]
        delta_P = P_CO2 - P_ETS                            # [€/tCO2]
    else:
        # Heat: carbon break-even not computed (no robust market reference)
        P_CO2   = None
        delta_P = None

    # Volumetric quantities — used in system-level aggregate (Section 5)
    cost_gap_total    = c.demand * price_gap               # [€/yr]
    co2_avoided_total = c.demand * c.e_av                  # [tCO2/yr]

    results.append({
        "carrier":          c.name,
        "unit":             c.unit,
        "carbon_analysis":  c.carbon_analysis,
        "demand":           c.demand,
        "P_sell":           c.P_sell,
        "P_mkt":            c.P_mkt,
        "e_av":             c.e_av,
        "price_gap":        price_gap,
        "P_CO2":            P_CO2,
        "delta_P":          delta_P,
        "cost_gap_total":   cost_gap_total,
        "co2_avoided_total": co2_avoided_total,
    })

# =============================================================================
# SECTION 5 — SYSTEM-LEVEL AGGREGATE (all three carriers)
# =============================================================================
#
# The system-level break-even is computed over all carriers. For heat, the
# natural gas boiler reference price and emission factor are used as an
# assumption, stated explicitly in the methodology.
#
#   P_CO2,sys = Σ_i [D_i (P_sell,i - P_market,i)]
#             / Σ_i [D_i * e_avoided,i]              [€/tCO2]
#
#   ΔP_sys    = P_CO2,sys - P_ETS                    [€/tCO2]

total_cost_gap    = sum(r["cost_gap_total"]    for r in results)   # [€/yr]
total_co2_avoided = sum(r["co2_avoided_total"] for r in results)   # [tCO2/yr]

# System-level break-even CO2 price (demand-weighted average across all carriers)
P_CO2_sys   = total_cost_gap / total_co2_avoided                   # [€/tCO2]

# System-level CO2 cost gap
delta_P_sys = P_CO2_sys - P_ETS                                    # [€/tCO2]

# =============================================================================
# SECTION 6 — PRINT RESULTS
# =============================================================================

SEP = "─" * 68

def fmt_co2(val):
    """Format a CO2 break-even price with ETS comparison."""
    if val is None:
        return "N/A (cost comparison only)"
    if val <= 0:
        return "Already competitive"
    flag = "✓ below ETS" if val <= P_ETS else f"✗ ETS gap: +€{val - P_ETS:.0f}/t"
    return f"€{val:.1f}/tCO₂  ({flag})"

def fmt_gap(val):
    """Format a CO2 cost gap (ΔP)."""
    if val is None:
        return "N/A"
    return f"€{val:+.1f}/tCO₂"

print(f"\n{SEP}")
print("  PER-CARRIER RESULTS")
print(SEP)
for r in results:
    print(f"\n  {r['carrier']}  {'[carbon break-even]' if r['carbon_analysis'] else '[cost comparison only]'}")
    print(f"    P_sell            : €{r['P_sell']:.2f}/{r['unit']}")
    print(f"    P_market          : €{r['P_mkt']:.2f}/{r['unit']}")
    print(f"    Price gap         : €{r['price_gap']:+.2f}/{r['unit']}")
    print(f"    Annual demand     : {r['demand']:>15,.0f} {r['unit']}/yr")
    print(f"    Total cost gap    : €{r['cost_gap_total']:>14,.0f}/yr")
    if r["carbon_analysis"]:
        print(f"    e_avoided         : {r['e_av']:.4f} tCO₂/{r['unit']}")
        print(f"    CO₂ avoided       : {r['co2_avoided_total']:>15,.1f} tCO₂/yr")
        print(f"    Break-even P_CO2  : {fmt_co2(r['P_CO2'])}")
        print(f"    ΔP (gap to ETS)   : {fmt_gap(r['delta_P'])}")

print(f"\n{SEP}")
print("  SYSTEM-LEVEL AGGREGATE  (electricity + H2 + heat)")
print("  Note: heat uses natural gas boiler reference as assumption")
print(SEP)
print(f"    Total CO₂ avoided  : {total_co2_avoided:>15,.1f} tCO₂/yr")
print(f"    Total cost gap     : €{total_cost_gap:>14,.0f}/yr")
print(f"    System P_CO2,sys   : {fmt_co2(P_CO2_sys)}")
print(f"    ΔP_sys (gap to ETS): {fmt_gap(delta_P_sys)}")
print(f"    EU ETS reference   : €{P_ETS:.0f}/tCO₂")
print(SEP)

# =============================================================================
# SECTION 7 — PLOTS
# =============================================================================

COLORS     = {"Electricity": "#378ADD", "Hydrogen": "#D85A30", "Heat": "#639922"}
be_values  = [r["P_CO2"] for r in results if r["P_CO2"] and r["P_CO2"] > 0]
MAX_X      = np.ceil(max(*be_values, P_CO2_sys, P_ETS) * 1.5 / 50) * 50
co2_range  = np.linspace(0, MAX_X, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Carbon break-even analysis", fontsize=13, fontweight="normal", y=1.01)

# ── Left panel: per-carrier (electricity and H2 only) ────────────────────────
# Conventional cost rises linearly with CO2 price; selling price is flat.
# Intersection = per-carrier break-even P_CO2.
for r in results:
    col = COLORS[r["carrier"]]
    if r["carbon_analysis"]:
        # Carbon break-even plot for electricity and H2
        conv_cost = r["P_mkt"] + r["e_av"] * co2_range    # P_market + e * P_CO2
        ax1.plot(co2_range, conv_cost, color=col, linewidth=2,
                 label=f"{r['carrier']} (conventional)")
        ax1.axhline(r["P_sell"], color=col, linewidth=1.5, linestyle="--",
                    label=f"{r['carrier']} (your price)")
        if 0 < r["P_CO2"] <= MAX_X:
            ax1.axvline(r["P_CO2"], color=col, linewidth=0.8,
                        linestyle=":", alpha=0.6)
    else:
        # Heat: show only as a horizontal cost gap annotation, no curve
        ax1.axhline(r["P_sell"], color=col, linewidth=1.5, linestyle="--",
                    label=f"{r['carrier']} (your price, no BE computed)")
        ax1.axhline(r["P_mkt"], color=col, linewidth=1.0, linestyle=":",
                    alpha=0.5, label=f"{r['carrier']} (market ref.)")

ax1.axvline(P_ETS, color="#888", linewidth=1.5, linestyle="--",
            label=f"EU ETS (€{P_ETS:.0f}/t)")
ax1.set_xlabel("CO₂ price (€/tCO₂)")
ax1.set_ylabel("Cost (€/unit of carrier)")
ax1.set_title("Per-carrier break-even\n(carbon analysis: electricity & H₂ only)")
ax1.legend(fontsize=8)
ax1.set_xlim(0, MAX_X)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}"))
ax1.grid(axis="both", linewidth=0.4, alpha=0.4)

# ── Right panel: system-level ─────────────────────────────────────────────────
# Total economic value of avoided CO2 (rising) vs flat total cost gap.
# Intersection = system-level break-even P_CO2,sys.
co2_value = co2_range * total_co2_avoided              # P_CO2 * Σ(D_i * e_i) [€/yr]

ax2.plot(co2_range, co2_value / 1e6, color="#7F77DD", linewidth=2,
         label="Value of avoided CO₂ (M€/yr)")
ax2.axhline(total_cost_gap / 1e6, color="#D85A30", linewidth=2, linestyle="--",
            label=f"Total cost gap (€{total_cost_gap/1e6:.1f}M/yr)")
ax2.axvline(P_ETS, color="#888", linewidth=1.5, linestyle="--",
            label=f"EU ETS (€{P_ETS:.0f}/t)")
if 0 < P_CO2_sys <= MAX_X:
    ax2.axvline(P_CO2_sys, color="#7F77DD", linewidth=1, linestyle=":",
                label=f"System break-even (€{P_CO2_sys:.0f}/t)")

ax2.set_xlabel("CO₂ price (€/tCO₂)")
ax2.set_ylabel("M€ / year")
ax2.set_title("System-level: cost gap vs CO₂ value\n(all carriers, heat via gas boiler assumption)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, MAX_X)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"€{y:.1f}M"))
ax2.grid(axis="both", linewidth=0.4, alpha=0.4)

fig.tight_layout()

output = Path(__file__).parent.parent / "[03] Figures" / "CarbonBreakEven" / "CarbonBreakEven.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
print(f"\nFigure saved → {output}")
plt.show()