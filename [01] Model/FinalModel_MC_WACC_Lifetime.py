'''Author: Gonçalo Costa Pina
Post-Optimisation Monte Carlo — WACC & Lifetime uncertainty only
Date_Created: 2026-05-19
Date_Modified: 2026-06-03

----------------------------------------------

Performs Monte Carlo analysis WITHOUT re-running the oemof optimiser.
Capacities are fixed at the base-case optimal values.
Only WACC (parameters.wacc) and system lifetime (parameters.general_lifetime)
are sampled; all CAPEX and OPEX terms are held at their base-case values.

Outputs:
  - Histogram grid  : LCOH2, LCOEl, LCOH, LCOE  (4-panel)
  - Tornado plots   : LCOH2, LCOEl, LCOH, LCOE  (one per carrier)
  - Tornado plot    : Total system cost & NPV    (combined panel)

Run from the [01] Model folder.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import truncnorm
from Parameters import parameters as par

# =============================================================================
# CONFIGURATION
# =============================================================================

# Baseline values for percentage conversion
BASE_TOTAL_COST_MEur = 1348179029.08053 / 1e6   # → 1348.18 M€/yr
BASE_NPV_MEur        = 2995060825.14534 / 1e6   # → 2995.06 M€

N_SAMPLES     = 100_000
RANDOM_SEED   = 42
SCENARIO_NAME = "MC_POST_OPT_WACC_LIFETIME"

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
    capex : capital cost per unit [€/MW or €/MWh]  – scalar or array
    n     : system lifetime [years]                 – scalar or array
    wacc  : discount rate [-]                       – scalar or array
    u     : component lifetime [years]; triggers replacement annuity if provided
    """
    crf_n = np.where(
        wacc == 0,
        1.0 / n,
        wacc * (1 + wacc) ** n / ((1 + wacc) ** n - 1),
    )

    if u is None:
        return capex * crf_n

    u     = float(u) if not np.ndim(u) else u
    n_arr = np.asarray(n, dtype=float)

    crf_u = np.where(
        wacc == 0,
        1.0 / u,
        wacc * (1 + wacc) ** u / ((1 + wacc) ** u - 1),
    )

    n_periods   = np.floor(n_arr / u).astype(int)
    max_periods = int(np.max(n_periods))

    discount_sum = np.ones_like(
        np.asarray(capex, dtype=float) * np.ones_like(wacc)
    )
    for k in range(1, max_periods):
        t    = k * u
        mask = k < n_periods
        discount_sum += np.where(mask, (1 + wacc) ** (-t), 0.0)

    return capex * crf_u * discount_sum * crf_n / crf_u


def _n(base, std_frac, lo_frac, hi_frac):
    """Truncated-normal spec: (baseline, abs_std, abs_min, abs_max).
    std_frac, lo_frac, hi_frac are expressed as fractions of base."""
    return (base, base * std_frac, base * lo_frac, base * hi_frac)


def sample_all(n):
    """Draw n samples from all truncated-normal parameter distributions."""
    s = {}
    for name, (base, std, lo, hi) in PARAM_DISTS.items():
        a = (lo - base) / std
        b = (hi - base) / std
        s[name] = truncnorm.rvs(
            a, b, loc=base, scale=std, size=n, random_state=rng,
        )
    return s


# =============================================================================
# LEVELISED COST CALCULATION  (vectorised)
# =============================================================================

def compute_lcox(s, n):
    """
    Compute levelised costs for all n samples in one vectorised pass.
    s must contain 'wacc' and 'lifetime'.
    Returns dict of result arrays (each length n).
    """
    w  = s["wacc"]
    lt = s["lifetime"]

    # ── Annualised ep_costs (CAPEX fixed at base-case par.* values) ───────────
    ep_pv = (
        annuity(par.PV_CAPEX_PANELS,   lt, w, u=par.PV_LIFETIME_SYSTEM)
        + annuity(par.PV_CAPEX_INVERTER, lt, w, u=par.PV_LIFETIME_INVERTER)
        + par.PV_OPEX
    )
    ep_wind = (
        annuity(par.WIND_CAPEX, lt, w, u=par.WIND_LIFETIME_SYSTEM)
        + par.WIND_OPEX
    )
    ep_electrolyser = (
        annuity(par.ELECTROLYSER_CAPEX_SYSTEM, lt, w, u=par.ELECTROLYSER_LIFETIME_SYSTEM)
        + annuity(par.ELECTROLYSER_CAPEX_STACK,  lt, w, u=par.ELECTROLYSER_LIFETIME_STACK)
        + par.ELECTROLYSER_OPEX
    )
    ep_heatpump = (
        annuity(par.HEATPUMP_CAPEX, lt, w, u=par.HEATPUMP_LIFETIME)
        + par.HEATPUMP_OPEX_FIX
    )
    ep_fuelcell = (
        annuity(par.FUELCELL_CAPEX_SYSTEM, lt, w, u=par.FUELCELL_LIFETIME_SYSTEM)
        + annuity(par.FUELCELL_CAPEX_STACK,  lt, w, u=par.FUELCELL_LIFETIME_STACK)
        + par.FUELCELL_OPEX_FIX
    )
    ep_battery_energy = (
        annuity(par.BATTERY_CAPEX_SYSTEM,        lt, w, u=par.BATTERY_LIFETIME_SYSTEM)
        + annuity(par.BATTERY_CAPEX_BATTERY_PACK,  lt, w, u=par.BATTERY_LIFETIME_BATTERY_PACK)
    )
    ep_battery_power  = par.BATTERY_OPEX_FIX
    ep_saltcavern_energy = (
        annuity(par.SALTCAVERN_CAPEX,              lt, w, u=par.SALTCAVERN_LIFETIME)
        + annuity(par.SALTCAVERN_CAPEX_CUSHION_GAS,  lt, w, u=par.SALTCAVERN_LIFETIME)
        + par.SALTCAVERN_OPEX
    )
    ep_saltcavern_power = annuity(
        par.SALTCAVERN_CAPEX_COMPRESSOR, lt, w, u=par.SALTCAVERN_LIFETIME_COMPRESSOR,
    )
    ep_thermalstorage = (
        annuity(par.THERMALSTORAGE_CAPEX, lt, w, u=par.THERMALSTORAGE_LIFETIME)
        + par.THERMALSTORAGE_OPEX
    )

    # ── Annual costs per technology ───────────────────────────────────────────
    cost_pv           = ep_pv            * CAP["pv"]
    cost_wind         = ep_wind          * CAP["wind"]
    cost_electrolyser = ep_electrolyser  * CAP["electrolyser"]
    cost_heatpump     = ep_heatpump      * CAP["heatpump"]
    cost_fuelcell     = ep_fuelcell      * CAP["fuelcell"]
    cost_battery      = (ep_battery_energy * CAP["battery_energy"]
                       + ep_battery_power  * CAP["battery_power"])
    cost_saltcavern   = (ep_saltcavern_energy * CAP["saltcavern_energy"]
                       + ep_saltcavern_power  * CAP["saltcavern_power"])
    cost_thermstor    = ep_thermalstorage * CAP["thermalstorage_energy"]

    # ── Variable OPEX (fixed at base-case operational values) ─────────────────
    var_hp  = par.HEATPUMP_OPEX_VAR          * HEATPUMP_INPUT_MWH
    var_bat = par.BATTERY_OPEX_VAR / 2       * (BAT_CHARGE_MWH + BAT_DISCHARGE_MWH)
    var_sc  = par.SALTCAVERN_OPEX_COMPRESSOR * SC_CHARGE_MWH

    # ── Total system cost (= oemof objective equivalent) ─────────────────────
    total_cost = (cost_pv + cost_wind + cost_electrolyser + cost_heatpump
                + cost_fuelcell + cost_battery + cost_saltcavern + cost_thermstor
                + var_hp + var_bat + var_sc)

    # ── Cost attribution per carrier ──────────────────────────────────────────
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
        "total_cost_eur_per_yr": total_cost,
    }


# =============================================================================
# NPV CALCULATION  (vectorised)
# =============================================================================

def compute_npv(s, n):
    """
    Reconstruct NPV for each sample, mirroring the main model's financial block.

    NPV  = -CAPEX_undiscounted
           + sum_{t=1}^{lifetime} (annual_revenue - annual_opex) / (1+wacc)^t

    where:
      annual_revenue = (LCOX_base * (1 + MARGIN)) * annual_demand  for each carrier
      annual_opex    = fixed_opex + variable_opex  (base-case operational values)

    Note: CAPEX_undiscounted is fixed (capacities & unit costs do not change);
    only the discount factor and horizon change with sampled wacc/lifetime.
    Lifetime is rounded to the nearest integer for the cash-flow sum.
    """
    w  = s["wacc"]
    lt = s["lifetime"]

    # ── Selling prices (main model: LC / (1 - margin)) ────────────────────────
    # Use base-case LC values (sanity-checked against main model below)
    h2_price = BASE_LCOH2 / (1.0 - par.MARGIN)
    el_price = BASE_LCOEL / (1.0 - par.MARGIN)
    ht_price = BASE_LCOH  / (1.0 - par.MARGIN)

    # ── Annual revenue (fixed demand, fixed selling price) ────────────────────
    annual_revenue = (h2_price * TOTAL_H2_MWH
                    + el_price * TOTAL_EL_MWH
                    + ht_price * TOTAL_HEAT_MWH)   # scalar €/yr

    # ── Annual OPEX (fixed OPEX rates × base-case capacities / flows) ─────────
    annual_fixed_opex = (
          CAP["pv"]                    * par.PV_OPEX
        + CAP["wind"]                  * par.WIND_OPEX
        + CAP["electrolyser"]          * par.ELECTROLYSER_OPEX
        + CAP["heatpump"]              * par.HEATPUMP_OPEX_FIX
        + CAP["fuelcell"]              * par.FUELCELL_OPEX_FIX
        + CAP["battery_power"]         * par.BATTERY_OPEX_FIX
        + CAP["saltcavern_energy"]     * par.SALTCAVERN_OPEX
        + CAP["thermalstorage_energy"] * par.THERMALSTORAGE_OPEX
    )
    annual_var_opex = (
          par.HEATPUMP_OPEX_VAR          * HEATPUMP_INPUT_MWH
        + par.BATTERY_OPEX_VAR           * (BAT_CHARGE_MWH + BAT_DISCHARGE_MWH) / 2
        + par.SALTCAVERN_OPEX_COMPRESSOR * SC_CHARGE_MWH
    )
    annual_opex = annual_fixed_opex + annual_var_opex   # scalar €/yr

    annual_cashflow = annual_revenue - annual_opex      # scalar €/yr

    # ── Undiscounted upfront CAPEX (scalar — capacities & unit costs fixed) ───
    capex_total = (
          CAP["pv"]                * (par.PV_CAPEX_PANELS + par.PV_CAPEX_INVERTER)
        + CAP["wind"]              *  par.WIND_CAPEX
        + CAP["electrolyser"]      *  par.ELECTROLYSER_CAPEX
        + CAP["heatpump"]          *  par.HEATPUMP_CAPEX
        + CAP["fuelcell"]          *  par.FUELCELL_CAPEX
        + CAP["battery_energy"]    *  par.BATTERY_CAPEX
        + CAP["battery_power"]     *  par.BATTERY_CAPEX_POWER
        + CAP["saltcavern_energy"] * (par.SALTCAVERN_CAPEX + par.SALTCAVERN_CAPEX_CUSHION_GAS)
        + CAP["saltcavern_power"]  *  par.SALTCAVERN_CAPEX_COMPRESSOR
        + CAP["thermalstorage_energy"] * par.THERMALSTORAGE_CAPEX
    )

    # ── Vectorised annuity of cash flows (present value of perpetuity finite) ──
    # PV of annual_cashflow over horizon lt at rate w:
    #   PV = annual_cashflow * [ (1 - (1+w)^-lt) / w ]   for w > 0
    #   PV = annual_cashflow * lt                          for w == 0
    lt_int = np.round(lt).astype(int)   # integer horizon per sample

    pv_cashflows = np.where(
        w == 0,
        annual_cashflow * lt_int,
        annual_cashflow * (1.0 - (1.0 + w) ** (-lt_int)) / w,
    )

    npv = -capex_total + pv_cashflows
    return npv


# =============================================================================
# TORNADO PLOT HELPERS
# =============================================================================

def _oat_sweep(metric_fn, base_samples, sensitivity=0.50):
    """
    One-at-a-time sweep: for each parameter in PARAM_DISTS, evaluate
    metric_fn at (base * (1-sensitivity)) and (base * (1+sensitivity)).

    metric_fn : callable(s) -> scalar, where s is the sample dict
    Returns list of dicts with keys param, delta_low, delta_high, total_swing.
    """
    mc_mean = float(np.mean(metric_fn(base_samples)))   # baseline reference

    rows = []
    for param_label, members in COMPONENT_GROUPS.items():
        lead      = members[0]
        base_val  = PARAM_DISTS[lead][0]
        if base_val == 0:
            continue

        s_low  = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}
        s_high = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}

        for member in members:
            member_base = PARAM_DISTS[member][0]
            if member_base == 0:
                continue
            s_low[member]  = np.array([member_base * (1.0 - sensitivity)])
            s_high[member] = np.array([member_base * (1.0 + sensitivity)])

        lc_low  = float(metric_fn(s_low))
        lc_high = float(metric_fn(s_high))

        rows.append({
            "param":       param_label,
            "delta_low":   lc_low  - mc_mean,
            "delta_high":  lc_high - mc_mean,
            "total_swing": abs(lc_high - lc_low),
        })

    return pd.DataFrame(rows).sort_values("total_swing", ascending=True)


def _draw_tornado(ax, tdf, sensitivity, xlabel):
    """Draw a single tornado chart onto ax from a pre-built tdf DataFrame."""
    n_rows = len(tdf)
    y_pos  = np.arange(n_rows)

    for i, row in enumerate(tdf.itertuples()):
        ax.barh(i, row.delta_high, left=0,
                color="steelblue", alpha=0.80, edgecolor="none", height=0.65)
        ax.barh(i, row.delta_low,  left=0,
                color="firebrick", alpha=0.80, edgecolor="none", height=0.65)

    all_deltas = tdf[["delta_low", "delta_high"]].values
    max_abs    = np.abs(all_deltas).max()
    ax.set_xlim(-(max_abs * 1.15), max_abs * 1.15)

    # ── Value labels ──────────────────────────────────────────────────────────
    text_objs = []
    for i, row in enumerate(tdf.itertuples()):
        tl = ax.text(row.delta_low  - 0.5, i, f"{row.delta_low:+.1f}",
                     va="center", ha="right", fontsize=13)
        tr = ax.text(row.delta_high + 0.5, i, f"{row.delta_high:+.1f}",
                     va="center", ha="left",  fontsize=13)
        text_objs.extend([tl, tr])

    # Expand xlim to fit labels
    plt.gcf().canvas.draw()
    renderer = plt.gcf().canvas.get_renderer()
    x_min, x_max = ax.get_xlim()
    for txt in text_objs:
        bb   = txt.get_window_extent(renderer=renderer)
        inv  = ax.transData.inverted()
        x0d, _ = inv.transform((bb.x0, bb.y0))
        x1d, _ = inv.transform((bb.x1, bb.y1))
        x_min = min(x_min, x0d)
        x_max = max(x_max, x1d)
    margin = 0.05 * (x_max - x_min)
    ax.set_xlim(x_min - margin, x_max + margin)

    ax.axvline(0, color="gray", lw=0.8, ls=":")
    legend_handles = [
        Patch(facecolor="steelblue", alpha=0.80, label=f"+{sensitivity*100:.0f}%"),
        Patch(facecolor="firebrick", alpha=0.80, label=f"\u2212{sensitivity*100:.0f}%"),
    ]
    ax.legend(handles=legend_handles, fontsize=13, loc="lower right")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tdf["param"], fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(axis="x", labelsize=13)


def tornado_plot_carrier(metric_key, label, filename, sensitivity=0.50):
    """
    OAT tornado for a single LCOX metric. x-axis = delta vs MC mean [€/MWh].
    """
    def metric_fn(s):
        return compute_lcox(s, 1)[metric_key]

    s_base = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}
    tdf    = _oat_sweep(metric_fn, s_base, sensitivity)

    n_rows = len(tdf)
    fig, ax = plt.subplots(figsize=(10, max(3, n_rows * 1.2)))
    _draw_tornado(
        ax, tdf, sensitivity,
        f"Change in {label} vs MC mean [€/MWh]  (\u00b1{sensitivity*100:.0f}% sensitivity)",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_{filename}.pdf", bbox_inches="tight")
    plt.show()


def _draw_tornado_pct(ax, tdf, sensitivity, metric_label, unit, panel_idx):
    """
    Like _draw_tornado but x-axis values are already % of baseline.
    Labels show '+X.X %' / '−X.X %'.
    metric_label : descriptive name inserted into the x-axis label.
    unit         : unit string shown in the x-axis label (e.g. 'M€/yr').
    """
    n_rows = len(tdf)
    y_pos  = np.arange(n_rows)

    for i, row in enumerate(tdf.itertuples()):
        ax.barh(i, row.delta_high, left=0,
                color="steelblue", alpha=0.80, edgecolor="none", height=0.55)
        ax.barh(i, row.delta_low,  left=0,
                color="firebrick", alpha=0.80, edgecolor="none", height=0.55)

    all_deltas = tdf[["delta_low", "delta_high"]].values
    max_abs    = np.abs(all_deltas).max()
    ax.set_xlim(-(max_abs * 1.20), max_abs * 1.20)

    # ── Value labels ──────────────────────────────────────────────────────────
    text_objs = []
    for i, row in enumerate(tdf.itertuples()):
        tl = ax.text(row.delta_low  - 0.3, i, f"{row.delta_low:+.1f}%",
                     va="center", ha="right", fontsize=12)
        tr = ax.text(row.delta_high + 0.3, i, f"{row.delta_high:+.1f}%",
                     va="center", ha="left",  fontsize=12)
        text_objs.extend([tl, tr])

    # Expand xlim to prevent label clipping
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_min, x_max = ax.get_xlim()
    for txt in text_objs:
        bb  = txt.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        x0d, _ = inv.transform((bb.x0, bb.y0))
        x1d, _ = inv.transform((bb.x1, bb.y1))
        x_min = min(x_min, x0d)
        x_max = max(x_max, x1d)
    margin = 0.05 * (x_max - x_min)
    ax.set_xlim(x_min - margin, x_max + margin)

    ax.axvline(0, color="gray", lw=0.8, ls=":")

    legend_handles = [
        Patch(facecolor="steelblue", alpha=0.80, label=f"+{sensitivity*100:.0f}%"),
        Patch(facecolor="firebrick", alpha=0.80, label=f"\u2212{sensitivity*100:.0f}%"),
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc="best")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tdf["param"], fontsize=12)
    ax.set_xlabel(
        f"Change in {metric_label} vs baseline / [%]",
        fontsize=12,
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.set_title("")   # ensure no title is shown

    ax.text(-0.07, 0.98, f"({chr(97 + panel_idx)})",
            transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", zorder=10)





def tornado_plot_system(sensitivity=0.50):
    """
    Combined 2-panel tornado (stacked vertically) for total system cost and NPV.
    x-axis shows percentage change relative to the known baseline values.
    No figure title; metric name is embedded in each panel's x-axis label.
    """
    s_base = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}

    def cost_fn(s):
        return compute_lcox(s, 1)["total_cost_eur_per_yr"] / 1e6   # M€/yr

    def npv_fn(s):
        return compute_npv(s, 1) / 1e6                              # M€

    tdf_cost = _oat_sweep(cost_fn, s_base, sensitivity)
    tdf_npv  = _oat_sweep(npv_fn,  s_base, sensitivity)

    # ── Convert deltas to % of baseline ──────────────────────────────────────
    for col in ("delta_low", "delta_high", "total_swing"):
        tdf_cost[col] = tdf_cost[col] / BASE_TOTAL_COST_MEur * 100
        tdf_npv[col]  = tdf_npv[col]  / BASE_NPV_MEur        * 100

    # ── Align row order (sorted by cost swing, largest impact at top) ─────────
    param_order = tdf_cost.sort_values("total_swing", ascending=True)["param"].tolist()
    tdf_cost = tdf_cost.set_index("param").loc[param_order].reset_index()
    tdf_npv  = tdf_npv.set_index("param").loc[param_order].reset_index()

    n_rows = len(tdf_cost)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 2 * n_rows + 1.5),
        constrained_layout=True,
    )

    for panel_idx, (ax, tdf, metric_label, unit) in enumerate(zip(
        axes,
        [tdf_cost,            tdf_npv],
        ["Annualised System Cost", "NPV"],
        ["M€/yr",             "M€"],
    )):
        _draw_tornado_pct(ax, tdf, sensitivity, metric_label, unit, panel_idx)

    fig.savefig(
        OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_tornado_system_cost_npv.pdf",
        bbox_inches="tight",
    )
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
# BASE-CASE CAPACITIES & OPERATIONAL VALUES  (paste from main model run)
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

# ── Base-case levelised costs (used as selling-price anchors for NPV) ─────────
# These must match main model outputs — verified in the sanity check below.
BASE_LCOH2 = 295.10   # €/MWh   (main model LCOH2)
BASE_LCOEL = 690.80   # €/MWh   (main model LCOEl)
BASE_LCOH  = 197.40   # €/MWh   (main model LCOH)


# =============================================================================
# PARAMETER DISTRIBUTIONS — WACC & LIFETIME ONLY
# Baseline values from logbook keys:
#   parameters.wacc             → par.WACC
#   parameters.general_lifetime → par.LIFETIME
# Each entry: (baseline, std_frac, lo_frac, hi_frac)
#   std_frac = 0.25  →  σ = 25 % of baseline
#   lo/hi_frac = 0.50 / 1.50  →  hard bounds at ±50 % of baseline
# =============================================================================

PARAM_DISTS = {
    "wacc":     _n(par.WACC,     0.25, 0.50, 1.50),   # parameters.wacc
    "lifetime": _n(par.LIFETIME, 0.25, 0.50, 1.50),   # parameters.general_lifetime
}

# OAT groups: one bar per uncertain parameter
COMPONENT_GROUPS = {
    "WACC":     ["wacc"],
    "Lifetime": ["lifetime"],
}


# =============================================================================
# RUN
# =============================================================================

print(f"Sampling {N_SAMPLES:,} parameter sets (WACC & lifetime uncertainty only)...")
samples = sample_all(N_SAMPLES)

print("Computing levelised costs (vectorised, no solver)...")
results = compute_lcox(samples, N_SAMPLES)

print("Computing NPV (vectorised)...")
npv_samples = compute_npv(samples, N_SAMPLES)

df = pd.DataFrame({
    **{f"param_{k}": v for k, v in samples.items()},
    **results,
    "npv_eur": npv_samples,
})

csv_path = OUTPUT_DIR / f"{SCENARIO_NAME}_results.csv"
df.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")


# =============================================================================
# SANITY CHECK — MC at baseline vs main model
# =============================================================================

s_base = {k: np.array([v[0]]) for k, v in PARAM_DISTS.items()}
r_base = compute_lcox(s_base, 1)
npv_base = compute_npv(s_base, 1)

print("\n" + "=" * 65)
print("  SANITY CHECK — MC at baseline vs main model")
print("=" * 65)
print(f"  LCOH2  MC baseline : {r_base['lcoh2_eur_per_mwh'][0]:>8.2f}  main model: {BASE_LCOH2:.2f}")
print(f"  LCOEl  MC baseline : {r_base['lcoel_eur_per_mwh'][0]:>8.2f}  main model: {BASE_LCOEL:.2f}")
print(f"  LCOH   MC baseline : {r_base['lcoh_eur_per_mwh'][0]:>8.2f}  main model: {BASE_LCOH:.2f}")
print(f"  LCOE   MC baseline : {r_base['lcoe_eur_per_mwh'][0]:>8.2f}  main model: 322.30")
print(f"  Total cost [M€/yr] : {r_base['total_cost_eur_per_yr'][0]/1e6:>8.2f}")
print(f"  NPV baseline [M€]  : {npv_base[0]/1e6:>8.2f}")
print("=" * 65)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

metrics = ["lcoh2_eur_per_mwh", "lcoh2_eur_per_kg",
           "lcoel_eur_per_mwh", "lcoh_eur_per_mwh", "lcoe_eur_per_mwh",
           "total_cost_eur_per_yr", "npv_eur"]

print("\n" + "=" * 78)
print(f"  POST-OPT MC (WACC & Lifetime only) — SUMMARY  (N={N_SAMPLES:,})")
print("=" * 78)
for m in metrics:
    col = df[m].dropna()
    print(f"  {m:<30}  mean={col.mean():>12.2f}  std={col.std():>10.2f}"
          f"  p5={col.quantile(0.05):>12.2f}  p95={col.quantile(0.95):>12.2f}")
print("=" * 78)


# =============================================================================
# PLOT 1 — Histogram grid  (LCOH2, LCOEl, LCOH, LCOE)
# =============================================================================

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
    ax.axvline(data.mean(), color="black", lw=1.5, ls="--",
               label=f"Average:\n{data.mean():.1f} €/MWh")
    ax.set_xlabel(label, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=13, loc="best")
    ax.annotate(
        f"({letter})",
        xy=(-0.02, 1.02), xycoords="axes fraction",
        fontsize=13, fontweight="bold", ha="right", va="bottom",
    )

global_ymax = max(ax.get_ylim()[1] for ax in axes.flat)
for ax in axes.flat:
    ax.set_ylim(0, global_ymax)

plt.tight_layout()
fig.savefig(OUTPUT_DIR_FIGS / f"{SCENARIO_NAME}_lc_histograms.pdf", bbox_inches="tight")
plt.show()


# =============================================================================
# PLOT 2 — OAT tornado plots — one per LCOX carrier
# =============================================================================

tornado_plot_carrier("lcoh2_eur_per_mwh", r"LCOH$_2$",                   "tornado_lcoh2")
tornado_plot_carrier("lcoel_eur_per_mwh", r"LCOE$_\mathrm{l}$",           "tornado_lcoel")
tornado_plot_carrier("lcoh_eur_per_mwh",  r"LCOH",                        "tornado_lcoh")
tornado_plot_carrier("lcoe_eur_per_mwh",  r"LCOE",                        "tornado_lcoe")


# =============================================================================
# PLOT 3 — Combined tornado: total system cost & NPV  (2-panel)
# =============================================================================

tornado_plot_system(sensitivity=0.50)


print("\nDone.")