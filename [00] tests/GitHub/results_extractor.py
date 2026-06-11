"""
results_extractor.py
====================
Extract and organize results from a solved Pyomo optimization model.

Functions
---------
extract_capacities   – optimal sizing [MW / MWh]
extract_dispatch     – hourly dispatch timeseries as a DataFrame
extract_costs        – cost breakdown and LCOH2
extract_all          – convenience wrapper returning a unified dict
validate_energy_balance – post-solve sanity checks
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import parameters as p


# ---------------------------------------------------------------------------
# Capacity results
# ---------------------------------------------------------------------------

def extract_capacities(model: pyo.ConcreteModel) -> dict[str, float]:
    """
    Return optimal installed capacities from the solved model.

    Returns
    -------
    dict with keys:
        ``pv_mw``, ``wind_mw``, ``electrolyser_mw``,
        ``battery_power_mw``, ``battery_energy_mwh``,
        ``cavern_mwh``
    """
    return {
        "pv_mw": pyo.value(model.X_pv),
        "wind_mw": pyo.value(model.X_wind),
        "electrolyser_mw": pyo.value(model.X_elec),
        "battery_power_mw": pyo.value(model.X_batt_power),
        "battery_energy_mwh": pyo.value(model.X_batt_energy),
        "cavern_mwh": pyo.value(model.X_cavern),
    }


# ---------------------------------------------------------------------------
# Dispatch timeseries
# ---------------------------------------------------------------------------

def extract_dispatch(model: pyo.ConcreteModel) -> pd.DataFrame:
    """
    Extract hourly dispatch variables from the solved model.

    Returns
    -------
    pd.DataFrame with columns:
        ``hour``, ``pv_gen``, ``wind_gen``, ``elec_in``,
        ``batt_charge``, ``batt_discharge``, ``batt_soc``,
        ``dump``, ``h2_produced``, ``cavern_charge``,
        ``cavern_discharge``, ``cavern_inv``, ``h2_demand``,
        ``elec_on``
    """
    n = len(model.T)
    rows = []
    for t in model.T:
        rows.append({
            "hour": t,
            "pv_gen": pyo.value(model.pv_cf[t]) * pyo.value(model.X_pv),
            "wind_gen": pyo.value(model.wind_cf[t]) * pyo.value(model.X_wind),
            "elec_in": pyo.value(model.elec_in[t]),
            "batt_charge": pyo.value(model.batt_charge[t]),
            "batt_discharge": pyo.value(model.batt_discharge[t]),
            "batt_soc": pyo.value(model.batt_soc[t]),
            "dump": pyo.value(model.dump[t]),
            "h2_produced": pyo.value(model.h2_produced[t]),
            "cavern_charge": pyo.value(model.cavern_charge[t]),
            "cavern_discharge": pyo.value(model.cavern_discharge[t]),
            "cavern_inv": pyo.value(model.cavern_inv[t]),
            "h2_demand": pyo.value(model.h2_demand[t]),
            "elec_on": pyo.value(model.elec_on[t]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

def extract_costs(
    model: pyo.ConcreteModel,
    params: dict | None = None,
) -> dict[str, float]:
    """
    Compute the annualized cost breakdown and LCOH2.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved model.
    params : dict, optional
        Parameter overrides (same format as ``parameters.PARAMS``).

    Returns
    -------
    dict with keys (all in €/year unless otherwise noted):
        ``pv_capex_annual``, ``pv_opex_annual``,
        ``wind_capex_annual``, ``wind_opex_annual``,
        ``elec_capex_annual``, ``elec_opex_annual``,
        ``batt_capex_annual``, ``batt_opex_annual``,
        ``cavern_capex_annual``, ``cavern_opex_annual``,
        ``total_annual_cost``,
        ``total_h2_demand_mwh``,
        ``lcoh2``  [€/MWh_H2]
    """
    cfg = dict(p.PARAMS)
    if params:
        cfg.update(params)

    X_pv = pyo.value(model.X_pv)
    X_wind = pyo.value(model.X_wind)
    X_elec = pyo.value(model.X_elec)
    X_batt_power = pyo.value(model.X_batt_power)
    X_batt_energy = pyo.value(model.X_batt_energy)
    X_cavern = pyo.value(model.X_cavern)
    batt_throughput = pyo.value(model.batt_throughput)

    pv_capex_annual = cfg["pv_capex"] * cfg["pv_crf"] * X_pv
    pv_opex_annual = cfg["pv_opex"] * X_pv

    wind_capex_annual = cfg["wind_capex"] * cfg["wind_crf"] * X_wind
    wind_opex_annual = cfg["wind_opex"] * X_wind

    elec_capex_annual = cfg["elec_capex"] * cfg["elec_crf"] * X_elec
    elec_opex_annual = cfg["elec_opex"] * X_elec

    batt_capex_annual = cfg["batt_capex_energy"] * cfg["batt_crf"] * X_batt_energy
    batt_opex_annual = (
        cfg["batt_opex_power"] * X_batt_power
        + cfg["batt_opex_variable"] * batt_throughput
    )

    cavern_capex_annual = cfg["cavern_capex_per_mwh"] * cfg["cavern_crf"] * X_cavern
    cavern_opex_annual = cfg["cavern_opex_per_mwh"] * X_cavern

    total_annual_cost = (
        pv_capex_annual + pv_opex_annual
        + wind_capex_annual + wind_opex_annual
        + elec_capex_annual + elec_opex_annual
        + batt_capex_annual + batt_opex_annual
        + cavern_capex_annual + cavern_opex_annual
    )

    total_h2_demand_mwh = sum(
        pyo.value(model.h2_demand[t]) for t in model.T
    )

    lcoh2 = total_annual_cost / total_h2_demand_mwh if total_h2_demand_mwh > 0 else float("inf")

    return {
        "pv_capex_annual": pv_capex_annual,
        "pv_opex_annual": pv_opex_annual,
        "wind_capex_annual": wind_capex_annual,
        "wind_opex_annual": wind_opex_annual,
        "elec_capex_annual": elec_capex_annual,
        "elec_opex_annual": elec_opex_annual,
        "batt_capex_annual": batt_capex_annual,
        "batt_opex_annual": batt_opex_annual,
        "cavern_capex_annual": cavern_capex_annual,
        "cavern_opex_annual": cavern_opex_annual,
        "total_annual_cost": total_annual_cost,
        "total_h2_demand_mwh": total_h2_demand_mwh,
        "lcoh2": lcoh2,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def extract_all(
    model: pyo.ConcreteModel,
    params: dict | None = None,
) -> dict:
    """
    Return a unified result dictionary containing capacities, costs and
    the full dispatch DataFrame.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved model.
    params : dict, optional
        Parameter overrides.

    Returns
    -------
    dict with keys ``'capacities'``, ``'costs'``, ``'dispatch'``.
    """
    return {
        "capacities": extract_capacities(model),
        "costs": extract_costs(model, params=params),
        "dispatch": extract_dispatch(model),
    }


# ---------------------------------------------------------------------------
# Energy-balance validation
# ---------------------------------------------------------------------------

def validate_energy_balance(
    results: dict,
    tol: float = 1e-3,
) -> bool:
    """
    Post-solve sanity checks on the extracted results.

    Checks
    ------
    1. Electricity balance: PV + Wind + Batt_discharge
                            = Elec_in + Batt_charge + Dump   (per hour)
    2. H2 balance: H2_produced + Cavern_discharge
                   = H2_demand + Cavern_charge               (per hour)
    3. Battery SOC non-negative everywhere.
    4. Cavern inventory non-negative everywhere.

    Parameters
    ----------
    results : dict
        Output of :func:`extract_all`.
    tol : float
        Absolute tolerance for balance checks (MWh).

    Returns
    -------
    bool
        True if all checks pass, False otherwise (with printed warnings).
    """
    df = results["dispatch"]
    ok = True

    # ---- Electricity balance ------------------------------------------------
    supply_elec = df["pv_gen"] + df["wind_gen"] + df["batt_discharge"]
    demand_elec = df["elec_in"] + df["batt_charge"] + df["dump"]
    elec_imbalance = (supply_elec - demand_elec).abs()
    if elec_imbalance.max() > tol:
        print(
            f"[validate] WARNING: Electricity balance violated! "
            f"Max imbalance = {elec_imbalance.max():.4f} MWh at hour "
            f"{elec_imbalance.idxmax()}."
        )
        ok = False
    else:
        print(f"[validate] Electricity balance OK (max error = {elec_imbalance.max():.6f} MWh).")

    # ---- H2 balance ---------------------------------------------------------
    supply_h2 = df["h2_produced"] + df["cavern_discharge"]
    demand_h2 = df["h2_demand"] + df["cavern_charge"]
    h2_imbalance = (supply_h2 - demand_h2).abs()
    if h2_imbalance.max() > tol:
        print(
            f"[validate] WARNING: H2 balance violated! "
            f"Max imbalance = {h2_imbalance.max():.4f} MWh at hour "
            f"{h2_imbalance.idxmax()}."
        )
        ok = False
    else:
        print(f"[validate] H2 balance OK (max error = {h2_imbalance.max():.6f} MWh).")

    # ---- Non-negativity checks ----------------------------------------------
    if (df["batt_soc"] < -tol).any():
        print("[validate] WARNING: Battery SOC is negative in some hours.")
        ok = False
    else:
        print("[validate] Battery SOC non-negative OK.")

    if (df["cavern_inv"] < -tol).any():
        print("[validate] WARNING: Cavern inventory is negative in some hours.")
        ok = False
    else:
        print("[validate] Cavern inventory non-negative OK.")

    return ok


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_results(
    results: dict,
    output_dir: str | Path = "output",
) -> None:
    """
    Save all results to CSV and JSON files in ``output_dir``.

    Files created
    -------------
    ``dispatch.csv``       – hourly dispatch timeseries
    ``capacities.json``    – optimal sizes
    ``costs.json``         – cost breakdown
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Dispatch timeseries
    dispatch_path = out / "dispatch.csv"
    results["dispatch"].to_csv(dispatch_path, index=False)
    print(f"[results] Dispatch saved to {dispatch_path}")

    # Capacities
    cap_path = out / "capacities.json"
    with open(cap_path, "w") as f:
        json.dump(results["capacities"], f, indent=2)
    print(f"[results] Capacities saved to {cap_path}")

    # Costs
    cost_path = out / "costs.json"
    with open(cost_path, "w") as f:
        json.dump(results["costs"], f, indent=2)
    print(f"[results] Costs saved to {cost_path}")
