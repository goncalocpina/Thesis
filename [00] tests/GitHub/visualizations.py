"""
visualizations.py
=================
Publication-quality visualizations for the H2 energy system model.

All functions accept the ``results`` dict produced by
``results_extractor.extract_all()`` and return ``matplotlib.figure.Figure``
objects, so they can be displayed interactively or saved.

Plots
-----
1. plot_monthly_energy_flows   – monthly aggregated bar chart
2. plot_sample_week_dispatch   – stacked area chart for one week
3. plot_electrolyser_ldc       – load duration curve of the electrolyser
4. plot_storage_soc            – full-year battery & cavern SoC
5. plot_cost_breakdown         – pie / bar chart of annual costs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for headless environments)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: derive a DatetimeIndex for the dispatch DataFrame
# ---------------------------------------------------------------------------

def _get_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return an hourly DatetimeIndex starting 1 Jan of a reference year."""
    return pd.date_range(start="2023-01-01", periods=len(df), freq="h")


# ---------------------------------------------------------------------------
# 1. Monthly aggregated energy flows
# ---------------------------------------------------------------------------

def plot_monthly_energy_flows(
    results: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Stacked bar chart of monthly electricity and H2 energy flows.

    Bars show:
    - Supply: PV generation, Wind generation, Battery discharge
    - Demand: Electrolyser in, Battery charge, Curtailment
    """
    df = results["dispatch"].copy()
    df.index = _get_time_index(df)

    monthly = df.resample("ME").sum()
    months = [d.strftime("%b") for d in monthly.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Monthly Energy Flows", fontsize=14, fontweight="bold")

    # ---- Electricity side --------------------------------------------------
    ax = axes[0]
    supply_cols = ["pv_gen", "wind_gen", "batt_discharge"]
    supply_labels = ["PV", "Wind", "Battery discharge"]
    supply_colors = ["#f9c74f", "#4cc9f0", "#90be6d"]

    demand_cols = ["elec_in", "batt_charge", "dump"]
    demand_labels = ["Electrolyser in", "Battery charge", "Curtailment"]
    demand_colors = ["#f3722c", "#577590", "#adb5bd"]

    _stacked_bars(ax, monthly, supply_cols, supply_labels, supply_colors,
                  months, title="Electricity [MWh/month]", side="left")
    _stacked_bars(ax, monthly, demand_cols, demand_labels, demand_colors,
                  months, title="Electricity [MWh/month]", side="right", offset=0.4)
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy [MWh]")
    ax.set_title("Electricity Balance")
    ax.legend(loc="upper left", fontsize=8)

    # ---- H2 side -----------------------------------------------------------
    ax2 = axes[1]
    h2_supply = ["h2_produced", "cavern_discharge"]
    h2_supply_labels = ["H2 produced", "Cavern discharge"]
    h2_supply_colors = ["#f3722c", "#277da1"]

    h2_demand = ["h2_demand", "cavern_charge"]
    h2_demand_labels = ["H2 demand", "Cavern charge"]
    h2_demand_colors = ["#43aa8b", "#f9844a"]

    _stacked_bars(ax2, monthly, h2_supply, h2_supply_labels, h2_supply_colors,
                  months, title="H2 [MWh/month]", side="left")
    _stacked_bars(ax2, monthly, h2_demand, h2_demand_labels, h2_demand_colors,
                  months, title="H2 [MWh/month]", side="right", offset=0.4)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Energy [MWh_H2]")
    ax2.set_title("H2 Balance")
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


def _stacked_bars(ax, monthly, cols, labels, colors, months, title, side, offset=0.0):
    """Helper: draw a grouped stacked bar series."""
    n = len(months)
    x = np.arange(n)
    width = 0.35
    bottom = np.zeros(n)
    for col, label, color in zip(cols, labels, colors):
        vals = monthly[col].values if col in monthly.columns else np.zeros(n)
        ax.bar(x + offset, vals, width, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += vals


# ---------------------------------------------------------------------------
# 2. Sample week dispatch (stacked area)
# ---------------------------------------------------------------------------

def plot_sample_week_dispatch(
    results: dict,
    week_start_day: int = 172,   # 21 June (summer, high PV)
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Stacked-area hourly dispatch chart for a sample week.

    Parameters
    ----------
    week_start_day : int
        Day-of-year for the start of the sample week (0-indexed).
    """
    df = results["dispatch"].copy()
    h_start = week_start_day * 24
    h_end = h_start + 7 * 24
    week = df.iloc[h_start:h_end].reset_index(drop=True)
    hours = week["hour"] if "hour" in week.columns else np.arange(len(week))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Sample Week Dispatch (day {week_start_day}–{week_start_day + 6})",
        fontsize=13, fontweight="bold",
    )

    # ---- Electricity -------------------------------------------------------
    ax1.stackplot(
        hours,
        week["pv_gen"], week["wind_gen"],
        labels=["PV", "Wind"],
        colors=["#f9c74f", "#4cc9f0"],
        alpha=0.85,
    )
    ax1.plot(hours, week["elec_in"], color="#f3722c", lw=1.5, label="Electrolyser in")
    ax1.plot(hours, week["batt_soc"] / week["batt_soc"].max() * week["elec_in"].max(),
             color="#90be6d", lw=1, ls="--", label="Batt SoC (scaled)")
    ax1.set_ylabel("Power [MW]")
    ax1.set_title("Electricity dispatch")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # ---- H2 ---------------------------------------------------------------
    ax2.stackplot(
        hours,
        week["h2_produced"],
        labels=["H2 produced"],
        colors=["#f3722c"],
        alpha=0.75,
    )
    ax2.plot(hours, week["h2_demand"], color="#277da1", lw=2, label="H2 demand")
    ax2.fill_between(hours, 0, week["cavern_discharge"], alpha=0.4,
                     color="#4361ee", label="Cavern discharge")
    ax2.fill_between(hours, 0, -week["cavern_charge"], alpha=0.4,
                     color="#f9844a", label="Cavern charge")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("H2 [MWh/h]")
    ax2.set_title("H2 dispatch")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Electrolyser load duration curve
# ---------------------------------------------------------------------------

def plot_electrolyser_ldc(
    results: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Load duration curve (LDC) for the electrolyser.

    Shows the fraction of hours the electrolyser operates at or above
    each power level (normalised to rated capacity).
    """
    df = results["dispatch"]
    cap = results["capacities"]
    x_elec = cap["electrolyser_mw"]

    if x_elec < 1e-6:
        print("[visualizations] Electrolyser capacity is zero; skipping LDC.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No electrolyser installed", ha="center", va="center")
        return fig

    elec_cf = (df["elec_in"] / x_elec).sort_values(ascending=False).values
    hours = np.arange(1, len(elec_cf) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(hours, 0, elec_cf, color="#f3722c", alpha=0.7)
    ax.axhline(
        results.get("params", {}).get("elec_min_load", 0.10),
        color="black", ls="--", lw=1, label="Minimum load",
    )
    ax.set_xlabel("Hours per year (sorted)")
    ax.set_ylabel("Load factor (fraction of rated capacity)")
    ax.set_title("Electrolyser Load Duration Curve")
    ax.set_xlim(0, len(elec_cf))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Full-year battery & cavern state of charge
# ---------------------------------------------------------------------------

def plot_storage_soc(
    results: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Full-year state-of-charge (SoC) timeseries for battery and cavern.
    """
    df = results["dispatch"]
    cap = results["capacities"]

    idx = _get_time_index(df)
    batt_soc = pd.Series(df["batt_soc"].values, index=idx)
    cavern_soc = pd.Series(df["cavern_inv"].values, index=idx)

    batt_cap = cap["battery_energy_mwh"]
    cavern_cap = cap["cavern_mwh"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Storage State of Charge – Full Year", fontsize=13, fontweight="bold")

    # Battery
    ax1.fill_between(idx, 0, batt_soc.values, color="#90be6d", alpha=0.7, label="Battery SoC")
    if batt_cap > 1e-6:
        ax1.axhline(batt_cap, color="gray", ls="--", lw=1, label=f"Capacity ({batt_cap:.1f} MWh)")
    ax1.set_ylabel("Energy [MWh]")
    ax1.set_title("Battery State of Charge")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Cavern
    ax2.fill_between(idx, 0, cavern_soc.values, color="#277da1", alpha=0.7, label="Cavern inventory")
    if cavern_cap > 1e-6:
        ax2.axhline(cavern_cap, color="gray", ls="--", lw=1,
                    label=f"Capacity ({cavern_cap:.1f} MWh)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("H2 [MWh]")
    ax2.set_title("Salt Cavern H2 Inventory")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Cost breakdown (bar chart)
# ---------------------------------------------------------------------------

def plot_cost_breakdown(
    results: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart showing annualized cost breakdown by component.
    """
    costs = results["costs"]

    components = [
        ("PV CAPEX", costs["pv_capex_annual"]),
        ("PV OPEX", costs["pv_opex_annual"]),
        ("Wind CAPEX", costs["wind_capex_annual"]),
        ("Wind OPEX", costs["wind_opex_annual"]),
        ("Electrolyser CAPEX", costs["elec_capex_annual"]),
        ("Electrolyser OPEX", costs["elec_opex_annual"]),
        ("Battery CAPEX", costs["batt_capex_annual"]),
        ("Battery OPEX", costs["batt_opex_annual"]),
        ("Cavern CAPEX", costs["cavern_capex_annual"]),
        ("Cavern OPEX", costs["cavern_opex_annual"]),
    ]
    labels = [c[0] for c in components]
    values = [c[1] / 1e6 for c in components]  # convert to M€

    colors = [
        "#f9c74f", "#f9c74f",
        "#4cc9f0", "#4cc9f0",
        "#f3722c", "#f3722c",
        "#90be6d", "#90be6d",
        "#277da1", "#277da1",
    ]
    hatches = ["", "///", "", "///", "", "///", "", "///", "", "///"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_xlabel("Annual Cost [M€/year]")
    ax.set_title(
        f"Annualized Cost Breakdown\n"
        f"Total: {costs['total_annual_cost'] / 1e6:.2f} M€/year  |  "
        f"LCOH2: {costs['lcoh2']:.2f} €/MWh"
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all plots
# ---------------------------------------------------------------------------

def save_all_plots(results: dict, output_dir: str | Path = "output/figures") -> None:
    """Save all standard visualization plots to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_monthly_energy_flows(results, output_path=out / "monthly_energy_flows.png")
    plot_sample_week_dispatch(results, output_path=out / "sample_week_dispatch.png")
    plot_electrolyser_ldc(results, output_path=out / "electrolyser_ldc.png")
    plot_storage_soc(results, output_path=out / "storage_soc.png")
    plot_cost_breakdown(results, output_path=out / "cost_breakdown.png")

    print(f"[visualizations] All plots saved to {out.resolve()}")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str | Path | None) -> None:
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[visualizations] Saved {p}")
    plt.close(fig)
