'''Author: Gonçalo Costa Pina
Post-Optimisation Monte Carlo — CAPEX uncertainty only
Date_Created: 2026-05-19
Date_Modified: 2026-05-21

----------------------------------------------

Performs Monte Carlo analysis WITHOUT re-running the oemof optimiser.
Capacities are fixed at the base-case optimal values.
Only CAPEX parameters are sampled.
Fixed and variable OPEX are held at their base-case values throughout.

Run from the [01] Model folder.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, truncnorm
from Parameters import parameters as par

# =============================================================================
# CONFIGURATION
# =============================================================================

N_SAMPLES     = 100_000
RANDOM_SEED   = 42
SCENARIO_NAME = "MC_POST_OPT_CAPEX"

DATA_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR.parent / "[01] Model" / "Logbooks" / "MonteCarloResults"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR_FIGS = DATA_DIR.parent / "[03] Figures" / "MonteCarloResults"
OUTPUT_DIR_FIGS.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)

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


# =============================================================================
# ANNUITY HELPER  (vectorised — exact replication of oemof.tools.economics.annuity)
# =============================================================================

def annuity(capex, n, wacc, u=None):
    """
    Exact vectorised replication of oemof.tools.economics.annuity().
    capex : capital cost per unit [€/MW or €/MWh]
    n     : system lifetime [years]
    wacc  : discount rate [-]
    u     : component lifetime [years]; triggers replacement annuity if provided
    """
    crf_n = np.where(
        wacc == 0,
        1.0 / n,
        wacc * (1 + wacc) ** n / ((1 + wacc) ** n - 1),
    )

    if u is None:
        return capex * crf_n

    u = float(u) if not np.ndim(u) else u
    n = float(n)

    crf_u = np.where(
        wacc == 0,
        1.0 / u,
        wacc * (1 + wacc) ** u / ((1 + wacc) ** u - 1),
    )

    n_periods   = int(np.floor(n / u)) if np.ndim(u) == 0 else np.floor(n / u).astype(int)
    max_periods = int(np.max(n_periods))

    discount_sum = np.ones_like(np.asarray(capex, dtype=float))
    for k in range(1, max_periods):
        t        = k * u
        mask     = k < n_periods
        discount = (1 + wacc) ** (-t)
        discount_sum += np.where(mask, discount, 0.0)

    return capex * crf_u * discount_sum * crf_n / crf_u


def _n(base, std_frac, lo_frac, hi_frac):
    """Normal distribution clipped to [base*lo_frac, base*hi_frac]."""
    return (base, base * std_frac, base * lo_frac, base * hi_frac)


def sample_all(n):
    """
    Draw n samples from all truncated-normal parameter distributions.
    """
    s = {}

    for name, (base, std, lo, hi) in PARAM_DISTS.items():

        # Convert absolute bounds into standard-normal coordinates
        a = (lo - base) / std
        b = (hi - base) / std

        s[name] = truncnorm.rvs(
            a, b,
            loc=base,
            scale=std,
            size=n,
            random_state=rng,
        )

    return s


# =============================================================================
# VECTORISED COST CALCULATION
# =============================================================================

def compute_lcox(s, n):
    """
    Compute levelised costs for all n samples in one vectorised pass.
    s : dict of sampled parameter arrays (each length n)
    Returns dict of result arrays (each length n).
    """
    w = par.WACC

    ep_pv = (
        annuity(s["pv_capex_panels"],   par.PV_LIFETIME_SYSTEM, w)
        + annuity(s["pv_capex_inverter"], par.PV_LIFETIME_SYSTEM, w,
                  u=par.PV_LIFETIME_INVERTER)
        + par.PV_OPEX
    )
    ep_wind = (
        annuity(s["wind_capex"], par.WIND_LIFETIME_SYSTEM, w)
        + par.WIND_OPEX
    )
    ep_electrolyser = (
        annuity(s["electrolyser_capex_system"], par.ELECTROLYSER_LIFETIME_SYSTEM, w)
        + annuity(s["electrolyser_capex_stack"], par.ELECTROLYSER_LIFETIME_SYSTEM, w,
                  u=par.ELECTROLYSER_LIFETIME_STACK)
        + par.ELECTROLYSER_OPEX
    )
    ep_heatpump = (
        annuity(s["heatpump_capex"], par.HEATPUMP_LIFETIME, w)
        + par.HEATPUMP_OPEX_FIX
    )
    ep_fuelcell = (
        annuity(s["fuelcell_capex_system"], par.FUELCELL_LIFETIME_SYSTEM, w)
        + annuity(s["fuelcell_capex_stack"], par.FUELCELL_LIFETIME_SYSTEM, w,
                  u=par.FUELCELL_LIFETIME_STACK)
        + par.FUELCELL_OPEX_FIX
    )
    ep_battery_energy = (
        annuity(s["battery_capex_system"],       par.LIFETIME, w, u=par.BATTERY_LIFETIME_SYSTEM)
        + annuity(s["battery_capex_battery_pack"], par.LIFETIME, w, u=par.BATTERY_LIFETIME_BATTERY_PACK)
    )
    ep_battery_power = ( par.BATTERY_OPEX_FIX )
    ep_saltcavern_energy = (
        annuity(s["saltcavern_capex"],             par.SALTCAVERN_LIFETIME, w)
        + annuity(s["saltcavern_capex_cushion_gas"], par.SALTCAVERN_LIFETIME, w)
        + par.SALTCAVERN_OPEX
    )
    ep_saltcavern_power = annuity(
        s["saltcavern_capex_compressor"],
        par.SALTCAVERN_LIFETIME, w,
        u=par.SALTCAVERN_LIFETIME_COMPRESSOR,
    )
    ep_thermalstorage = (
        annuity(par.THERMALSTORAGE_CAPEX, par.THERMALSTORAGE_LIFETIME, w)
        + par.THERMALSTORAGE_OPEX
    )

    cost_pv           = ep_pv           * CAP["pv"]
    cost_wind         = ep_wind         * CAP["wind"]
    cost_electrolyser = ep_electrolyser * CAP["electrolyser"]
    cost_heatpump     = ep_heatpump     * CAP["heatpump"]
    cost_fuelcell     = ep_fuelcell     * CAP["fuelcell"]
    cost_battery      = (ep_battery_energy * CAP["battery_energy"]
                       + ep_battery_power  * CAP["battery_power"])
    cost_saltcavern   = (ep_saltcavern_energy * CAP["saltcavern_energy"]
                       + ep_saltcavern_power  * CAP["saltcavern_power"])
    cost_thermstor    = ep_thermalstorage * CAP["thermalstorage_energy"]

    var_hp  = par.HEATPUMP_OPEX_VAR          * HEATPUMP_INPUT_MWH
    var_bat = par.BATTERY_OPEX_VAR / 2       * (BAT_CHARGE_MWH + BAT_DISCHARGE_MWH)
    var_sc  = par.SALTCAVERN_OPEX_COMPRESSOR * SC_CHARGE_MWH

    total_energy = TOTAL_H2_MWH + TOTAL_EL_MWH + TOTAL_HEAT_MWH
    frac_h2   = TOTAL_H2_MWH   / total_energy
    frac_el   = TOTAL_EL_MWH   / total_energy
    frac_heat = TOTAL_HEAT_MWH / total_energy

    cost_supply  = cost_pv + cost_wind
    cost_el_h2   = cost_electrolyser * ELECTRO_FRAC_H2
    cost_el_heat = cost_electrolyser * ELECTRO_FRAC_HEAT
    cost_fc_el   = cost_fuelcell * FC_FRAC_EL
    cost_fc_heat = cost_fuelcell * FC_FRAC_HEAT

    cost_h2_total   = cost_supply * frac_h2   + cost_el_h2   + cost_saltcavern + var_sc
    cost_el_total   = cost_supply * frac_el   + cost_battery  + var_bat + cost_fc_el
    cost_heat_total = (cost_supply * frac_heat + cost_heatpump + var_hp
                     + cost_thermstor + cost_el_heat + cost_fc_heat)
    total_cost      = cost_h2_total + cost_el_total + cost_heat_total

    lcoh2 = cost_h2_total   / TOTAL_H2_MWH   if TOTAL_H2_MWH   > 0 else np.full(n, np.nan)
    lcoel = cost_el_total   / TOTAL_EL_MWH   if TOTAL_EL_MWH   > 0 else np.full(n, np.nan)
    lcoh  = cost_heat_total / TOTAL_HEAT_MWH if TOTAL_HEAT_MWH > 0 else np.full(n, np.nan)
    lcoe  = total_cost      / total_energy   if total_energy    > 0 else np.full(n, np.nan)

    return {
        "lcoh2_eur_per_mwh": lcoh2,
        "lcoh2_eur_per_kg":  lcoh2 * par.H2_CALORIFIC_VALUE_LHV,
        "lcoel_eur_per_mwh": lcoel,
        "lcoh_eur_per_mwh":  lcoh,
        "lcoe_eur_per_mwh":  lcoe,
        "total_cost_eur":    total_cost,
    }


# =============================================================================
# GROUP COST DECOMPOSITION
# =============================================================================

def compute_group_costs(s, n):
    """Annual cost [M€/yr] broken down into Supply, PtX (incl. heat pump), Storage."""
    w = par.WACC

    ep_pv = (
        annuity(s["pv_capex_panels"],   par.PV_LIFETIME_SYSTEM, w)
        + annuity(s["pv_capex_inverter"], par.PV_LIFETIME_SYSTEM, w,
                  u=par.PV_LIFETIME_INVERTER)
        + par.PV_OPEX
    )
    ep_wind = annuity(s["wind_capex"], par.WIND_LIFETIME_SYSTEM, w) + par.WIND_OPEX
    cost_supply = ep_pv * CAP["pv"] + ep_wind * CAP["wind"]

    ep_electrolyser = (
        annuity(s["electrolyser_capex_system"], par.ELECTROLYSER_LIFETIME_SYSTEM, w)
        + annuity(s["electrolyser_capex_stack"], par.ELECTROLYSER_LIFETIME_SYSTEM, w,
                  u=par.ELECTROLYSER_LIFETIME_STACK)
        + par.ELECTROLYSER_OPEX
    )
    ep_fuelcell = (
        annuity(s["fuelcell_capex_system"], par.FUELCELL_LIFETIME_SYSTEM, w)
        + annuity(s["fuelcell_capex_stack"], par.FUELCELL_LIFETIME_SYSTEM, w,
                  u=par.FUELCELL_LIFETIME_STACK)
        + par.FUELCELL_OPEX_FIX
    )
    ep_heatpump = (
        annuity(s["heatpump_capex"], par.HEATPUMP_LIFETIME, w)
        + par.HEATPUMP_OPEX_FIX
    )
    cost_ptx = (
          ep_electrolyser * CAP["electrolyser"]
        + ep_fuelcell     * CAP["fuelcell"]
        + ep_heatpump     * CAP["heatpump"]
    )

    ep_battery_energy = (
        annuity(s["battery_capex_system"],        par.LIFETIME, w, u=par.BATTERY_LIFETIME_SYSTEM)
        + annuity(s["battery_capex_battery_pack"], par.LIFETIME, w, u=par.BATTERY_LIFETIME_BATTERY_PACK)
    )
    ep_saltcavern_energy = (
        annuity(s["saltcavern_capex"],              par.SALTCAVERN_LIFETIME, w)
        + annuity(s["saltcavern_capex_cushion_gas"], par.SALTCAVERN_LIFETIME, w)
        + par.SALTCAVERN_OPEX
    )
    ep_saltcavern_power = annuity(
        s["saltcavern_capex_compressor"], par.SALTCAVERN_LIFETIME, w,
        u=par.SALTCAVERN_LIFETIME_COMPRESSOR,
    )
    cost_storage = (
          ep_battery_energy    * CAP["battery_energy"]
        + par.BATTERY_OPEX_FIX * CAP["battery_power"]
        + ep_saltcavern_energy * CAP["saltcavern_energy"]
        + ep_saltcavern_power  * CAP["saltcavern_power"]
    )

    return pd.DataFrame({
        "Supply":  cost_supply  / 1e6,
        "PtX":     cost_ptx     / 1e6,
        "Storage": cost_storage / 1e6,
    })


# =============================================================================
# TORNADO PLOT HELPER
# =============================================================================

def tornado_plot_carrier(metric_key, base_value, label, filename, sensitivity=0.50):
    """
    One-at-a-time tornado plot, one bar per component group.
    x=0 is the MC distribution mean; bars show delta from that mean.
    sensitivity: fractional variation applied to each parameter (default ±50%).
    """
    mc_mean    = df[metric_key].mean()
    base_delta = base_value - mc_mean

    rows = []
    for group_label, members in COMPONENT_GROUPS.items():
        lead = members[0]
        base_val = PARAM_DISTS[lead][0]
        if base_val == 0:
            continue

        frac_low  = 1.0 - sensitivity
        frac_high = 1.0 + sensitivity

        s_low  = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}
        s_high = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}

        for member in members:
            member_base = PARAM_DISTS[member][0]
            if member_base == 0:
                continue
            s_low[member]  = np.array([member_base * frac_low])
            s_high[member] = np.array([member_base * frac_high])

        lc_low  = compute_lcox(s_low,  1)[metric_key][0]
        lc_high = compute_lcox(s_high, 1)[metric_key][0]

        rows.append({
            "param":       group_label,
            "delta_low":   lc_low  - mc_mean,
            "delta_high":  lc_high - mc_mean,
            "total_swing": abs(lc_high - lc_low),
        })

    tdf = pd.DataFrame(rows).sort_values("total_swing", ascending=True)

    n_rows = len(tdf)
    fig, ax = plt.subplots(figsize=(10, max(5, n_rows * 0.9)))
    y_pos = np.arange(n_rows)

    # ── Bars ──────────────────────────────────────────────────────────────────
    for i, row in enumerate(tdf.itertuples()):
        ax.barh(i, row.delta_high, left=0,
                color="steelblue", alpha=0.80, edgecolor="none", height=0.65)
        ax.barh(i, row.delta_low,  left=0,
                color="firebrick", alpha=0.80, edgecolor="none", height=0.65)

    # Set a provisional xlim so text objects are placed in a sensible coordinate
    # space before we ask the renderer to measure them
    all_deltas = tdf[["delta_low", "delta_high"]].values
    max_abs    = np.abs(all_deltas).max()
    ax.set_xlim(-(max_abs * 1.15), max_abs * 1.15)

    # ── Labels (draw first, measure second) ───────────────────────────────────
    text_objects = []
    for i, row in enumerate(tdf.itertuples()):
        tl = ax.text(row.delta_low  - 0.5, i, f"{row.delta_low:+.1f}",
                     va="center", ha="right", fontsize=14)
        tr = ax.text(row.delta_high + 0.5, i, f"{row.delta_high:+.1f}",
                     va="center", ha="left",  fontsize=14)
        text_objects.extend([tl, tr])

    # ── Renderer-based xlim expansion ─────────────────────────────────────────
    fig.canvas.draw()                          # forces layout & text placement
    renderer = fig.canvas.get_renderer()

    x_min, x_max = ax.get_xlim()
    for txt in text_objects:
        bb = txt.get_window_extent(renderer=renderer)
        # Convert from display (pixel) coords back to data coords
        inv = ax.transData.inverted()
        x0_data, _ = inv.transform((bb.x0, bb.y0))
        x1_data, _ = inv.transform((bb.x1, bb.y1))
        x_min = min(x_min, x0_data)
        x_max = max(x_max, x1_data)

    margin = 0.05 * (x_max - x_min)           # 5% breathing room on each side
    ax.set_xlim(x_min - margin, x_max + margin)

    # ── Reference line & legend ───────────────────────────────────────────────
    ax.axvline(0, color="gray", lw=0.8, ls=":")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue", alpha=0.80, label=f"+{sensitivity*100:.0f}%"),
        Patch(facecolor="firebrick", alpha=0.80, label=f"−{sensitivity*100:.0f}%"),
    ]
    ax.legend(handles=legend_handles, fontsize=14, loc="lower right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tdf["param"], fontsize=14)
    ax.set_xlabel(
        f"Change in {label} vs MC mean [€/MWh]  (±{sensitivity*100:.0f}% sensitivity)",
        fontsize=14,
    )
    ax.tick_params(axis="x", labelsize=14)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_{filename}.pdf", bbox_inches="tight")
    plt.show()

# =============================================================================
# DATA IMPORT
# =============================================================================

data_path   = DATA_DIR.parent / "[02] Data" / "Supply"
demand_file = DATA_DIR.parent / "[02] Data" / "Demand" / "Demand_Profiles.csv"

df_pv = pd.read_csv(data_path / "ninja_pv_51.9244_4.4778_corrected.csv",
                    sep=",", skiprows=3)
df_pv = df_pv[["time", "electricity"]].copy()
df_pv["electricity"] = pd.to_numeric(df_pv["electricity"], errors="coerce")
df_pv = df_pv.dropna(subset=["electricity"])
df_pv["time"] = pd.to_datetime(df_pv["time"], format="%Y-%m-%d %H:%M")
df_pv = df_pv.sort_values("time")
print("PV profile loaded:", len(df_pv), "rows")

df_wind = pd.read_csv(data_path / "ninja_wind_51.9244_4.4778_corrected.csv",
                      sep=",", skiprows=3)
df_wind = df_wind[["time", "electricity"]].copy()
df_wind["electricity"] = pd.to_numeric(df_wind["electricity"], errors="coerce")
df_wind = df_wind.dropna(subset=["electricity"])
df_wind["time"] = pd.to_datetime(df_wind["time"], format="%Y-%m-%d %H:%M")
df_wind = df_wind.sort_values("time")
print("Wind profile loaded:", len(df_wind), "rows")

df_dem = pd.read_csv(demand_file, sep=";")
df_dem["Datetime (UTC)"] = pd.to_datetime(
    df_dem["Datetime (UTC)"], format="%d/%m/%Y %H:%M")
df_dem = df_dem[[
    "Datetime (UTC)",
    "Rotterdam_total_gas_demand [MW]",
    "Rotterdam_electricity_load [MW]",
    "Rotterdam_total_heat_demand [MW]",
]].copy()
for col in df_dem.columns[1:]:
    df_dem[col] = pd.to_numeric(df_dem[col], errors="coerce")
df_dem = df_dem.dropna().sort_values("Datetime (UTC)").reset_index(drop=True)
print("Demand profiles loaded:", len(df_dem), "rows")

timeseries = pd.DataFrame({
    "PV":                 df_pv["electricity"].values,
    "Wind":               df_wind["electricity"].values,
    "H2_Demand":          df_dem["Rotterdam_total_gas_demand [MW]"].values,
    "Electricity_Demand": df_dem["Rotterdam_electricity_load [MW]"].values,
    "Heat_Demand":        df_dem["Rotterdam_total_heat_demand [MW]"].values,
}, index=df_dem["Datetime (UTC)"])
timeseries.index = pd.DatetimeIndex(timeseries.index).round("h")
timeseries.index.name = "time"
timeseries = timeseries.asfreq("h")


# =============================================================================
# BASE-CASE RESULTS  ← paste your values here after running the main model
# =============================================================================

CAP = {
    "pv":                    2127.3,   # MW
    "wind":                  2534.7,   # MW
    "electrolyser":          681.1,    # MW
    "fuelcell":              10.3,     # MW
    "heatpump":              199.6,    # MW
    "battery_energy":        6820.8,   # MWh
    "battery_power":         1705.2,   # MW
    "saltcavern_energy":     67863.4,  # MWh
    "saltcavern_power":      559.1,    # MW
    "thermalstorage_energy": 0.0,      # MWh
    "thermalstorage_power":  0.0,      # MW
}

TOTAL_H2_MWH   = timeseries["H2_Demand"].sum()
TOTAL_EL_MWH   = timeseries["Electricity_Demand"].sum()
TOTAL_HEAT_MWH = timeseries["Heat_Demand"].sum()

HEATPUMP_INPUT_MWH = 342088.9   # heatpump_input.sum()
BAT_CHARGE_MWH     = 238895.4   # bat_charge.sum()
BAT_DISCHARGE_MWH  = 211797.5   # bat_discharge.sum()
SC_CHARGE_MWH      = 170824.4   # saltcavern_charge.sum()

_el_total         = par.ELECTROLYSER_EFFICIENCY + par.ELECTROLYSER_RECOVERABLE_HEAT
ELECTRO_FRAC_H2   = par.ELECTROLYSER_EFFICIENCY       / _el_total
ELECTRO_FRAC_HEAT = par.ELECTROLYSER_RECOVERABLE_HEAT / _el_total

_fc_total    = par.FUELCELL_EFFICIENCY_ELECTRICITY + par.FUELCELL_EFFICIENCY_HEAT
FC_FRAC_EL   = par.FUELCELL_EFFICIENCY_ELECTRICITY / _fc_total
FC_FRAC_HEAT = par.FUELCELL_EFFICIENCY_HEAT        / _fc_total


# =============================================================================
# PARAMETER DISTRIBUTIONS — CAPEX ONLY
# Each entry: (baseline, abs_std, abs_min, abs_max)
# =============================================================================

PARAM_DISTS = {
    # ── PV ────────────────────────────────────────────────────────────────────
    "pv_capex_panels":   _n(par.PV_CAPEX_PANELS,   0.25, 0.50, 1.50),
    "pv_capex_inverter": _n(par.PV_CAPEX_INVERTER, 0.25, 0.50, 1.50),
    # ── Wind ──────────────────────────────────────────────────────────────────
    "wind_capex": _n(par.WIND_CAPEX, 0.25, 0.50, 1.50),
    # ── Electrolyser ──────────────────────────────────────────────────────────
    "electrolyser_capex_system": _n(par.ELECTROLYSER_CAPEX_SYSTEM, 0.25, 0.50, 1.50),
    "electrolyser_capex_stack":  _n(par.ELECTROLYSER_CAPEX_STACK,  0.25, 0.50, 1.50),
    # ── Fuel Cell ─────────────────────────────────────────────────────────────
    "fuelcell_capex_system": _n(par.FUELCELL_CAPEX_SYSTEM, 0.25, 0.50, 1.50),
    "fuelcell_capex_stack":  _n(par.FUELCELL_CAPEX_STACK,  0.25, 0.50, 1.50),
    # ── Battery ───────────────────────────────────────────────────────────────
    "battery_capex_system":       _n(par.BATTERY_CAPEX_SYSTEM,       0.25, 0.50, 1.50),
    "battery_capex_battery_pack": _n(par.BATTERY_CAPEX_BATTERY_PACK, 0.25, 0.50, 1.50),
    # ── Salt Cavern ───────────────────────────────────────────────────────────
    "saltcavern_capex":             _n(par.SALTCAVERN_CAPEX,             0.25, 0.50, 1.50),
    "saltcavern_capex_cushion_gas": _n(par.SALTCAVERN_CAPEX_CUSHION_GAS, 0.25, 0.50, 1.50),
    "saltcavern_capex_compressor":  _n(par.SALTCAVERN_CAPEX_COMPRESSOR,  0.25, 0.50, 1.50),
    # ── Heat Pump ─────────────────────────────────────────────────────────────
    "heatpump_capex": _n(par.HEATPUMP_CAPEX, 0.25, 0.50, 1.50),
}

# Groups for tornado: one bar per component, sub-parameters perturbed together
COMPONENT_GROUPS = {
    "PV CAPEX":           ["pv_capex_panels", "pv_capex_inverter"],
    "Wind CAPEX":         ["wind_capex"],
    "Electrolyser CAPEX": ["electrolyser_capex_system", "electrolyser_capex_stack"],
    "Fuel Cell CAPEX":    ["fuelcell_capex_system", "fuelcell_capex_stack"],
    "Battery CAPEX":      ["battery_capex_system", "battery_capex_battery_pack"],
    "Salt Cavern CAPEX":  ["saltcavern_capex", "saltcavern_capex_cushion_gas",
                           "saltcavern_capex_compressor"],
    "Heat Pump CAPEX":    ["heatpump_capex"],
}


# =============================================================================
# RUN
# =============================================================================

print(f"Sampling {N_SAMPLES:,} parameter sets (CAPEX uncertainty only)...")
samples = sample_all(N_SAMPLES)

print("Computing levelised costs (vectorised, no solver)...")
results = compute_lcox(samples, N_SAMPLES)

df = pd.DataFrame({**{f"param_{k}": v for k, v in samples.items()}, **results})
csv_path = OUTPUT_DIR / f"{SCENARIO_NAME}_results.csv"
df.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")

# ── compute group costs here so gdf/gdf_base are available for all plots ──────
print("Computing group cost decomposition...")
gdf      = compute_group_costs(samples, N_SAMPLES)
s_base   = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}
gdf_base = compute_group_costs(s_base, 1)


# =============================================================================
# SANITY CHECK
# =============================================================================

r_base = compute_lcox(s_base, 1)

print("\n" + "=" * 65)
print("  SANITY CHECK — MC at baseline vs main model")
print("=" * 65)
print(f"  LCOH2  MC baseline : {r_base['lcoh2_eur_per_mwh'][0]:>8.2f}  main model: 295.10")
print(f"  LCOEl  MC baseline : {r_base['lcoel_eur_per_mwh'][0]:>8.2f}  main model: 690.80")
print(f"  LCOH   MC baseline : {r_base['lcoh_eur_per_mwh'][0]:>8.2f}  main model: 197.40")
print(f"  LCOE   MC baseline : {r_base['lcoe_eur_per_mwh'][0]:>8.2f}  main model: 322.30")
print("=" * 65)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

metrics = ["lcoh2_eur_per_mwh", "lcoh2_eur_per_kg",
           "lcoel_eur_per_mwh", "lcoh_eur_per_mwh", "lcoe_eur_per_mwh"]

print("\n" + "=" * 70)
print(f"  POST-OPTIMISATION MC (CAPEX only) — SUMMARY  (N={N_SAMPLES:,})")
print("=" * 70)
for m in metrics:
    col = df[m].dropna()
    print(f"  {m:<30}  mean={col.mean():>8.2f}  std={col.std():>7.2f}"
          f"  p5={col.quantile(0.05):>8.2f}  p95={col.quantile(0.95):>8.2f}")
print("=" * 70)


# =============================================================================
# PLOTS
# =============================================================================

# ── 1. Histogram grid ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_specs = [
    ("lcoh2_eur_per_mwh", r"LCOH$_2$ / [€/MWh]",           "tab:green"),
    ("lcoel_eur_per_mwh", r"LCOE$_\mathrm{l}$ / [€/MWh]",  "tab:olive"),
    ("lcoh_eur_per_mwh",  r"LCOH / [€/MWh]",                "tab:orange"),
    ("lcoe_eur_per_mwh",  r"LCOE / [€/MWh]",                "slateblue"),
]

for (ax, (col, label, color)), letter in zip(zip(axes.flat, plot_specs), "abcd"):
    data = df[col].dropna()
    ax.hist(data, bins=60, color=color, alpha=0.80, edgecolor="none")
    ax.axvline(data.mean(), color="black", lw=1.5, ls="--", label=f"Average: \n{data.mean():.1f} €/MWh")
    ax.set_xlabel(label, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=13, loc="best")

    ax.annotate(
        f"({letter})",
        xy=(-0.02, 1.02),          # just outside top-left corner
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

global_ymax = max(ax.get_ylim()[1] for ax in axes.flat)
for ax in axes.flat:
    ax.set_ylim(0, global_ymax)

plt.tight_layout()
fig.savefig(OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_lc_histograms.pdf", bbox_inches="tight")
plt.show()


# ── 2. OAT tornado plots — one per carrier, one bar per component ─────────────

tornado_plot_carrier("lcoh2_eur_per_mwh", 295.1,  r"LCOH$_2$", "tornado_lcoh2")
tornado_plot_carrier("lcoel_eur_per_mwh", 690.8,  r"LCOE$_l$", "tornado_lcoel")
tornado_plot_carrier("lcoh_eur_per_mwh",  197.4,  r"LCOH",     "tornado_lcoh")
tornado_plot_carrier("lcoe_eur_per_mwh",  322.30, r"LCOE",     "tornado_lcoe")


# ── 3. Scatter plots: group total annualised cost vs LCOE ─────────────────────
# x-axis = total annualised cost of the group [M€/yr] from gdf,
# aggregating all members (PV+Wind, Electrolyser+FC+HP, Battery+Cavern)

scatter_specs = [
    ("Supply",  "steelblue", "Supply annualised cost [M€/yr]\n(PV + Wind)",                         "lcoe_vs_supply_capex"),
    ("PtX",     "firebrick", "PtX annualised cost [M€/yr]\n(Electrolyser + Fuel cell + Heat pump)",  "lcoe_vs_ptx_capex"),
    ("Storage", "seagreen",  "Storage annualised cost [M€/yr]\n(Battery + Salt cavern)",             "lcoe_vs_storage_capex"),
]

for group, color, x_label, filename in scatter_specs:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        gdf[group],
        df["lcoe_eur_per_mwh"],
        s=1, alpha=0.15, color=color,
    )
    ax.axvline(gdf_base[group].iloc[0], color="black", lw=1.0, ls="--",
               label=f"Base case: {gdf_base[group].iloc[0]:.1f} M€/yr")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"LCOE [€/MWh]")
    ax.set_title(f"LCOE vs {group} annualised cost")
    #ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_{filename}.pdf", bbox_inches="tight")
    plt.show()


# ── Combined scatter plot: annualised cost groups vs LCOE ────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

scatter_specs = [
    ("Supply",  "steelblue", "Supply"),
    ("PtX",     "firebrick", "PtX"),
    ("Storage", "seagreen",  "Storage"),
]

for group, color, label in scatter_specs:

    ax.scatter(
        gdf[group],
        df["lcoe_eur_per_mwh"],
        s=2,
        alpha=0.15,
        color=color,
        label=label,
    )

    # Base-case vertical line
    '''ax.axvline(
        gdf_base[group].iloc[0],
        color=color,
        lw=1.2,
        ls="--",
        alpha=0.9,
    )'''

ax.set_xlabel("Annualised cost [M€/yr]")
ax.set_ylabel(r"LCOE [€/MWh]")

'''ax.set_title(
    "LCOE vs annualised system cost groups\n"
    "(Supply, PtX, Storage)"
)'''

leg = ax.legend(markerscale=4)
for handle in leg.legend_handles:
    handle.set_alpha(1)
ax.set_axisbelow(True)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_lcoe_vs_all_groups.pdf",
    bbox_inches="tight"
)

plt.show()

# ── 4. Violin plots: cost decomposition by sector ─────────────────────────────

groups       = ["Supply", "PtX", "Storage"]
group_colors = {"Supply": "steelblue", "PtX": "firebrick", "Storage": "seagreen"}

gdf_melt = gdf.melt(var_name="Group", value_name="Annual Cost Contribution [M€/yr]")
gdf_melt["Group"] = pd.Categorical(gdf_melt["Group"], categories=groups, ordered=True)

fig, ax = plt.subplots(1, 1, figsize=(16, 5))

sns.violinplot(
    data=gdf_melt,
    x="Group",
    y="Annual Cost Contribution [M€/yr]",
    ax=ax,
    hue="Group",
    palette=group_colors,
    cut=0,
    inner="box",
)



ax.set_xlabel("Group", fontsize=18)
ax.set_ylabel("Annual Cost Contribution [M€/yr]", fontsize=18)
ax.set_xlabel("", fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_axisbelow(True)
ax.tick_params(axis="x", direction="out", labelsize=18)


plt.tight_layout()
fig.savefig(OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_cost_decomposition.pdf", bbox_inches="tight")
plt.show()


print("\nDone.")