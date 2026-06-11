"""
sensitivity_analysis.py
=======================
Parameter sweep and sensitivity analysis for the H2 energy system model.

Each function varies one or more parameters over a user-defined range,
re-solves the optimization model, and collects key metrics.  Results are
returned as pandas DataFrames and optionally plotted as heatmaps.

Sweep functions
---------------
sweep_pv_wind_capex          – 2-D grid: PV CAPEX vs. Wind CAPEX
sweep_electrolyser_efficiency – 1-D sweep of η_elec
sweep_cavern_capex            – 1-D sweep of cavern CAPEX
sweep_battery_efficiency      – 1-D sweep of battery round-trip efficiency
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import parameters as p
from data_loader import generate_synthetic_data
from pyomo_model import build_model, solve_model
from results_extractor import extract_capacities, extract_costs


# ---------------------------------------------------------------------------
# Internal: run one optimization and return scalar metrics
# ---------------------------------------------------------------------------

def _run_one(
    profiles: dict,
    params: dict,
    solver: str = "cbc",
    mip_gap: float = 0.01,
    time_limit_seconds: int = 600,
) -> dict | None:
    """
    Build, solve, and extract key metrics for a given parameter set.

    Returns None if the solver did not find an optimal solution.
    """
    model = build_model(
        pv_cf=profiles["pv"],
        wind_cf=profiles["wind"],
        h2_demand=profiles["demand"],
        params=params,
    )
    model, status = solve_model(
        model,
        solver_name=solver,
        tee=False,
        mip_gap=mip_gap,
        time_limit_seconds=time_limit_seconds,
    )
    if not status["optimal"]:
        return None

    caps = extract_capacities(model)
    costs = extract_costs(model, params=params)
    return {**caps, **costs}


# ---------------------------------------------------------------------------
# 1. PV & Wind CAPEX 2-D sweep
# ---------------------------------------------------------------------------

def sweep_pv_wind_capex(
    pv_capex_range: list | None = None,
    wind_capex_range: list | None = None,
    n_points: int = 5,
    solver: str = "cbc",
    profiles: dict | None = None,
    mip_gap: float = 0.01,
    time_limit_seconds: int = 600,
) -> pd.DataFrame:
    """
    Sweep PV CAPEX and Wind CAPEX on a 2-D grid and record LCOH2 and
    optimal capacities.

    Parameters
    ----------
    pv_capex_range : list of two floats [min, max], optional
        PV CAPEX range in €/MW.  Default: [300 000, 900 000].
    wind_capex_range : list of two floats [min, max], optional
        Wind CAPEX range in €/MW.  Default: [800 000, 1 600 000].
    n_points : int
        Number of grid points along each axis.
    solver : str
        Pyomo solver name.
    profiles : dict, optional
        Pre-loaded ``{'pv', 'wind', 'demand'}`` arrays.  Synthetic data used
        if None.
    mip_gap : float
        Relative MIP optimality gap tolerance passed to the solver.
    time_limit_seconds : int
        Per-solve wall-clock time limit in seconds.

    Returns
    -------
    pd.DataFrame with columns:
        ``pv_capex``, ``wind_capex``, ``lcoh2``, ``pv_mw``, ``wind_mw``, …
    """
    if pv_capex_range is None:
        pv_capex_range = [300_000, 900_000]
    if wind_capex_range is None:
        wind_capex_range = [800_000, 1_600_000]

    pv_values = np.linspace(*pv_capex_range, n_points)
    wind_values = np.linspace(*wind_capex_range, n_points)

    if profiles is None:
        profiles = generate_synthetic_data()

    records = []
    total = n_points * n_points
    done = 0
    for pv_c in pv_values:
        for wind_c in wind_values:
            done += 1
            print(f"  [{done}/{total}] PV CAPEX={pv_c/1e3:.0f} k€/MW, "
                  f"Wind CAPEX={wind_c/1e3:.0f} k€/MW")
            params = dict(p.PARAMS)
            params["pv_capex"] = pv_c
            params["wind_capex"] = wind_c
            params["pv_crf"] = p.crf(params["discount_rate"], params["pv_lifetime"])
            params["wind_crf"] = p.crf(params["discount_rate"], params["wind_lifetime"])

            result = _run_one(profiles, params, solver, mip_gap, time_limit_seconds)
            if result is not None:
                result["pv_capex"] = pv_c
                result["wind_capex"] = wind_c
                records.append(result)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Electrolyser efficiency sweep
# ---------------------------------------------------------------------------

def sweep_electrolyser_efficiency(
    eta_range: list | None = None,
    n_points: int = 8,
    solver: str = "cbc",
    profiles: dict | None = None,
    mip_gap: float = 0.01,
    time_limit_seconds: int = 600,
) -> pd.DataFrame:
    """
    1-D sweep of electrolyser efficiency (LHV basis).

    Parameters
    ----------
    eta_range : [min, max], optional
        Efficiency range [-].  Default: [0.55, 0.80].

    Returns
    -------
    pd.DataFrame
    """
    if eta_range is None:
        eta_range = [0.55, 0.80]

    eta_values = np.linspace(*eta_range, n_points)
    if profiles is None:
        profiles = generate_synthetic_data()

    records = []
    for eta in eta_values:
        print(f"  Electrolyser η = {eta:.3f}")
        params = dict(p.PARAMS)
        params["elec_efficiency"] = eta

        result = _run_one(profiles, params, solver, mip_gap, time_limit_seconds)
        if result is not None:
            result["elec_efficiency"] = eta
            records.append(result)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Salt cavern CAPEX sweep
# ---------------------------------------------------------------------------

def sweep_cavern_capex(
    capex_range: list | None = None,
    n_points: int = 8,
    solver: str = "cbc",
    profiles: dict | None = None,
    mip_gap: float = 0.01,
    time_limit_seconds: int = 600,
) -> pd.DataFrame:
    """
    1-D sweep of salt cavern CAPEX (geology cost in €/kgH2).

    Parameters
    ----------
    capex_range : [min, max], optional
        Geology CAPEX range in €/kgH2.  Default: [10, 40].

    Returns
    -------
    pd.DataFrame
    """
    if capex_range is None:
        capex_range = [10.0, 40.0]

    capex_values = np.linspace(*capex_range, n_points)
    if profiles is None:
        profiles = generate_synthetic_data()

    records = []
    for cap_geo in capex_values:
        print(f"  Cavern CAPEX geology = {cap_geo:.1f} €/kgH2")
        params = dict(p.PARAMS)
        # Recompute total CAPEX/OPEX per MWh
        cap_per_kg = (
            cap_geo
            + p.CAVERN_CAPEX_CUSHION_GAS * (1 + p.CAVERN_CUSHION_GAS_FRACTION)
        )
        h2_lwh_mwh_per_kg = p.H2_LHV_MWH_PER_KG
        params["cavern_capex_per_mwh"] = cap_per_kg / h2_lwh_mwh_per_kg
        params["cavern_opex_per_mwh"] = (
            cap_geo / h2_lwh_mwh_per_kg * p.CAVERN_OPEX_FRACTION
        )

        result = _run_one(profiles, params, solver, mip_gap, time_limit_seconds)
        if result is not None:
            result["cavern_capex_geology_eur_per_kg"] = cap_geo
            records.append(result)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Battery round-trip efficiency sweep
# ---------------------------------------------------------------------------

def sweep_battery_efficiency(
    rt_range: list | None = None,
    n_points: int = 7,
    solver: str = "cbc",
    profiles: dict | None = None,
    mip_gap: float = 0.01,
    time_limit_seconds: int = 600,
) -> pd.DataFrame:
    """
    1-D sweep of battery round-trip efficiency.

    Parameters
    ----------
    rt_range : [min, max], optional
        RT efficiency range [-].  Default: [0.75, 0.97].

    Returns
    -------
    pd.DataFrame
    """
    if rt_range is None:
        rt_range = [0.75, 0.97]

    rt_values = np.linspace(*rt_range, n_points)
    if profiles is None:
        profiles = generate_synthetic_data()

    records = []
    for rt in rt_values:
        eta = np.sqrt(rt)  # symmetric charge/discharge efficiency
        print(f"  Battery RT efficiency = {rt:.3f} (η_in = η_out = {eta:.4f})")
        params = dict(p.PARAMS)
        params["batt_rt_efficiency"] = rt
        params["batt_eta_in"] = eta
        params["batt_eta_out"] = eta

        result = _run_one(profiles, params, solver, mip_gap, time_limit_seconds)
        if result is not None:
            result["batt_rt_efficiency"] = rt
            records.append(result)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Heatmap plotting helpers
# ---------------------------------------------------------------------------

def plot_heatmap_pv_wind(
    df: pd.DataFrame,
    metric: str = "lcoh2",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot a 2-D heatmap of ``metric`` from a PV/Wind CAPEX sweep DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`sweep_pv_wind_capex`.
    metric : str
        Column name to colour by (e.g. ``'lcoh2'``, ``'pv_mw'``, ``'wind_mw'``).
    """
    pivot = df.pivot_table(
        index="wind_capex",
        columns="pv_capex",
        values=metric,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="viridis_r" if "lcoh" in metric else "viridis")
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v/1e3:.0f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v/1e3:.0f}" for v in pivot.index])
    ax.set_xlabel("PV CAPEX [k€/MW]")
    ax.set_ylabel("Wind CAPEX [k€/MW]")
    ax.set_title(f"Sensitivity: {metric}\n(PV & Wind CAPEX sweep)")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color="white", fontsize=8)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_line_sweep(
    df: pd.DataFrame,
    x_col: str,
    metrics: list[str] | None = None,
    xlabel: str = "",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Line plot for 1-D parameter sweeps.

    Parameters
    ----------
    df : pd.DataFrame
        Output of any 1-D sweep function.
    x_col : str
        Column to use as the x-axis.
    metrics : list of str, optional
        Columns to plot.  Defaults to ``['lcoh2', 'pv_mw', 'wind_mw', 'electrolyser_mw']``.
    """
    if metrics is None:
        metrics = ["lcoh2", "pv_mw", "wind_mw", "electrolyser_mw"]
    metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(df[x_col], df[metric], marker="o", color="#3a86ff")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel or x_col)
    fig.suptitle(f"Sensitivity: {x_col}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def run_all_sweeps(
    solver: str = "cbc",
    n_points: int = 4,
    output_dir: str | Path = "output/sensitivity",
    mip_gap: float = 0.01,
    time_limit_seconds: int = 600,
) -> dict[str, pd.DataFrame]:
    """
    Run all four sensitivity sweeps, save results to CSV, and plot heatmaps.

    Parameters
    ----------
    solver : str
        Solver to use for each sub-problem.
    n_points : int
        Grid resolution (lower = faster).
    output_dir : str or Path
        Directory for CSV and figure output.
    mip_gap : float
        Relative MIP optimality gap tolerance.
    time_limit_seconds : int
        Per-solve wall-clock time limit in seconds.

    Returns
    -------
    dict mapping sweep name to result DataFrame.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    profiles = generate_synthetic_data()

    results = {}

    # 1. PV & Wind CAPEX heatmap
    print("\n=== Sweep 1: PV & Wind CAPEX ===")
    df_pv_wind = sweep_pv_wind_capex(
        n_points=n_points, solver=solver, profiles=profiles,
        mip_gap=mip_gap, time_limit_seconds=time_limit_seconds,
    )
    df_pv_wind.to_csv(out / "sweep_pv_wind_capex.csv", index=False)
    if not df_pv_wind.empty:
        plot_heatmap_pv_wind(df_pv_wind, metric="lcoh2",
                             output_path=out / "heatmap_pv_wind_lcoh2.png")
        plot_heatmap_pv_wind(df_pv_wind, metric="pv_mw",
                             output_path=out / "heatmap_pv_wind_pv_capacity.png")
        plot_heatmap_pv_wind(df_pv_wind, metric="wind_mw",
                             output_path=out / "heatmap_pv_wind_wind_capacity.png")
    results["pv_wind"] = df_pv_wind

    # 2. Electrolyser efficiency
    print("\n=== Sweep 2: Electrolyser Efficiency ===")
    df_elec = sweep_electrolyser_efficiency(
        n_points=n_points, solver=solver, profiles=profiles,
        mip_gap=mip_gap, time_limit_seconds=time_limit_seconds,
    )
    df_elec.to_csv(out / "sweep_elec_efficiency.csv", index=False)
    if not df_elec.empty:
        plot_line_sweep(
            df_elec, "elec_efficiency",
            metrics=["lcoh2", "electrolyser_mw", "pv_mw"],
            xlabel="Electrolyser Efficiency [-]",
            output_path=out / "sweep_elec_efficiency.png",
        )
    results["elec_efficiency"] = df_elec

    # 3. Salt cavern CAPEX
    print("\n=== Sweep 3: Salt Cavern CAPEX ===")
    df_cavern = sweep_cavern_capex(
        n_points=n_points, solver=solver, profiles=profiles,
        mip_gap=mip_gap, time_limit_seconds=time_limit_seconds,
    )
    df_cavern.to_csv(out / "sweep_cavern_capex.csv", index=False)
    if not df_cavern.empty:
        plot_line_sweep(
            df_cavern, "cavern_capex_geology_eur_per_kg",
            metrics=["lcoh2", "cavern_mwh", "pv_mw"],
            xlabel="Cavern CAPEX geology [€/kgH2]",
            output_path=out / "sweep_cavern_capex.png",
        )
    results["cavern_capex"] = df_cavern

    # 4. Battery efficiency
    print("\n=== Sweep 4: Battery Round-Trip Efficiency ===")
    df_batt = sweep_battery_efficiency(
        n_points=n_points, solver=solver, profiles=profiles,
        mip_gap=mip_gap, time_limit_seconds=time_limit_seconds,
    )
    df_batt.to_csv(out / "sweep_battery_efficiency.csv", index=False)
    if not df_batt.empty:
        plot_line_sweep(
            df_batt, "batt_rt_efficiency",
            metrics=["lcoh2", "battery_energy_mwh", "pv_mw"],
            xlabel="Battery RT Efficiency [-]",
            output_path=out / "sweep_battery_efficiency.png",
        )
    results["battery_efficiency"] = df_batt

    print(f"\n[sensitivity] All results saved to {out.resolve()}")
    return results


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path) -> None:
    if path is not None:
        p_path = Path(path)
        p_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p_path, dpi=150, bbox_inches="tight")
        print(f"[sensitivity] Saved {p_path}")
    plt.close(fig)