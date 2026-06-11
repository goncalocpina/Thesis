'''Author: Gonçalo Costa Pina
Monte Carlo Sensitivity Analysis
Date_Created: 2026-05-19
Date_Modified: 2026-05-21 

----------------------------------------------

Performs Monte Carlo analysis on the oemof energy system model by:
- Sampling uncertain CAPEX, lifetime, efficiency, and WACC parameters
- Re-solving the full investment optimisation for each sample
- Collecting LCOH2, LCOEl, LCOH, LCOE, optimal capacities, and total cost
- Saving all results to CSV and producing summary plots

Run from the [01] Model folder (same location as the main model script).

RUNTIME ESTIMATE: ~N × (single-run time). Start with N_SAMPLES = 50 to verify,
then scale up. Each oemof solve on a full year typically takes 1–5 min with CBC.
'''

import pandas as pd
import numpy as np
from pathlib import Path
from oemof import solph
from oemof.tools import economics
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import traceback
from tqdm import tqdm

from Parameters import parameters as par

# =============================================================================
# CONFIGURATION
# =============================================================================

N_SAMPLES   = 200       # number of MC iterations
RANDOM_SEED = 42        # for reproducibility (set None to disable)
SCENARIO_NAME = "MC_FINAL_MODEL"

# Output file
DATA_DIR    = Path(__file__).resolve().parent
OUTPUT_DIR  = DATA_DIR.parent / "[03] Figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / f"{SCENARIO_NAME}_mc_results.csv"

if RANDOM_SEED is not None:
    rng = np.random.default_rng(RANDOM_SEED)
else:
    rng = np.random.default_rng()


# =============================================================================
# PARAMETER DISTRIBUTIONS
# =============================================================================
# Each entry: (baseline, std, min_clip, max_clip)
# All CAPEX in €/MW or €/MWh as per your Parameters file.
# std expressed as fraction of baseline (e.g. 0.25 = ±25%).
# Lifetimes in years, WACC as decimal.

def _normal(baseline, std_frac, lo_frac, hi_frac):
    """Returns (baseline, abs_std, abs_min, abs_max)."""
    std = baseline * std_frac
    return baseline, std, baseline * lo_frac, baseline * hi_frac

PARAM_DISTS = {
    # ── Financial ─────────────────────────────────────────────────────────────
    "wacc": (
        par.WACC,
        0.02,                       # abs std: ±2 percentage points
        max(0.02, par.WACC - 0.05), # min
        par.WACC + 0.06,            # max
    ),

    # ── PV ────────────────────────────────────────────────────────────────────
    "pv_capex_panels":   _normal(par.PV_CAPEX_PANELS,   0.25, 0.50, 1.60),
    "pv_capex_inverter": _normal(par.PV_CAPEX_INVERTER, 0.20, 0.60, 1.50),

    # ── Wind ──────────────────────────────────────────────────────────────────
    "wind_capex": _normal(par.WIND_CAPEX, 0.25, 0.50, 1.60),

    # ── Electrolyser ──────────────────────────────────────────────────────────
    "electrolyser_capex_system": _normal(par.ELECTROLYSER_CAPEX_SYSTEM, 0.30, 0.40, 1.80),
    "electrolyser_capex_stack":  _normal(par.ELECTROLYSER_CAPEX_STACK,  0.30, 0.40, 1.80),
    "electrolyser_lifetime_stack": (
        par.ELECTROLYSER_LIFETIME_STACK,
        1.5,                                        # abs std: ±1.5 yr
        max(2, par.ELECTROLYSER_LIFETIME_STACK - 3),
        par.ELECTROLYSER_LIFETIME_STACK + 5,
    ),

    # ── Fuel Cell ─────────────────────────────────────────────────────────────
    "fuelcell_capex_system": _normal(par.FUELCELL_CAPEX_SYSTEM, 0.30, 0.40, 1.80),
    "fuelcell_capex_stack":  _normal(par.FUELCELL_CAPEX_STACK,  0.35, 0.40, 1.80),
    "fuelcell_lifetime_stack": (
        par.FUELCELL_LIFETIME_STACK,
        1.5,
        max(2, par.FUELCELL_LIFETIME_STACK - 3),
        par.FUELCELL_LIFETIME_STACK + 5,
    ),

    # ── Battery ───────────────────────────────────────────────────────────────
    "battery_capex_system":       _normal(par.BATTERY_CAPEX_SYSTEM,       0.25, 0.50, 1.60),
    "battery_capex_battery_pack": _normal(par.BATTERY_CAPEX_BATTERY_PACK, 0.25, 0.40, 1.70),
    "battery_capex_power":        _normal(par.BATTERY_CAPEX_POWER,        0.25, 0.50, 1.60),

    # ── Salt Cavern ───────────────────────────────────────────────────────────
    "saltcavern_capex":            _normal(par.SALTCAVERN_CAPEX,            0.40, 0.30, 2.00),
    "saltcavern_capex_cushion_gas":_normal(par.SALTCAVERN_CAPEX_CUSHION_GAS,0.40, 0.30, 2.00),
    "saltcavern_capex_compressor": _normal(par.SALTCAVERN_CAPEX_COMPRESSOR, 0.30, 0.50, 1.70),
}


def sample_parameters():
    """Draw one sample from all parameter distributions. Returns a dict."""
    s = {}
    for name, (base, std, lo, hi) in PARAM_DISTS.items():
        val = rng.normal(base, std)
        s[name] = float(np.clip(val, lo, hi))
    return s


# =============================================================================
# DATA LOADING  (done once, outside the loop)
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

df_wind = pd.read_csv(data_path / "ninja_wind_51.9244_4.4778_corrected.csv",
                      sep=",", skiprows=3)
df_wind = df_wind[["time", "electricity"]].copy()
df_wind["electricity"] = pd.to_numeric(df_wind["electricity"], errors="coerce")
df_wind = df_wind.dropna(subset=["electricity"])
df_wind["time"] = pd.to_datetime(df_wind["time"], format="%Y-%m-%d %H:%M")
df_wind = df_wind.sort_values("time")

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

# COP time series (unchanged — no uncertainty on COP here)
months   = timeseries.index.month
cop_ts   = np.where(
    (months >= par.HEATPUMP_SUMMERTIME_START) & (months <= par.HEATPUMP_SUMMERTIME_END),
    par.HEATPUMP_COP * par.HEATPUMP_COP_SUMMERFACTOR,
    par.HEATPUMP_COP * par.HEATPUMP_COP_WINTERFACTOR,
)

print("Data loaded successfully.")
print(f"Running {N_SAMPLES} Monte Carlo samples...\n")


# =============================================================================
# SINGLE-RUN FUNCTION
# =============================================================================

def run_single(p: dict) -> dict | None:
    """
    Build and solve the oemof model with sampled parameters p.
    Returns a flat dict of scalar results, or None if solve fails.
    """
    wacc = p["wacc"]

    # ── ep_costs (annualised CAPEX + fixed OPEX per MW or MWh) ───────────────
    ep_pv = (
        economics.annuity(capex=p["pv_capex_panels"],   n=par.PV_LIFETIME_SYSTEM, wacc=wacc)
        + economics.annuity(capex=p["pv_capex_inverter"], n=par.PV_LIFETIME_SYSTEM, wacc=wacc,
                            u=par.PV_LIFETIME_INVERTER)
        + par.PV_OPEX
    )
    ep_wind = (
        economics.annuity(capex=p["wind_capex"], n=par.WIND_LIFETIME_SYSTEM, wacc=wacc)
        + par.WIND_OPEX
    )
    ep_electrolyser = (
        economics.annuity(capex=p["electrolyser_capex_system"],
                          n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=wacc)
        + economics.annuity(capex=p["electrolyser_capex_stack"],
                            n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=wacc,
                            u=p["electrolyser_lifetime_stack"])
        + par.ELECTROLYSER_OPEX
    )
    ep_heatpump = (
        economics.annuity(capex=par.HEATPUMP_CAPEX, n=par.HEATPUMP_LIFETIME, wacc=wacc)
        + par.HEATPUMP_OPEX_FIX
    )
    ep_fuelcell = (
        economics.annuity(capex=p["fuelcell_capex_system"],
                          n=par.FUELCELL_LIFETIME_SYSTEM, wacc=wacc)
        + economics.annuity(capex=p["fuelcell_capex_stack"],
                            n=par.FUELCELL_LIFETIME_SYSTEM, wacc=wacc,
                            u=p["fuelcell_lifetime_stack"])
        + par.FUELCELL_OPEX_FIX
    )
    ep_battery_energy = (
        economics.annuity(capex=p["battery_capex_system"],
                          n=par.LIFETIME, wacc=wacc, u=par.BATTERY_LIFETIME_SYSTEM)
        + economics.annuity(capex=p["battery_capex_battery_pack"],
                            n=par.LIFETIME, wacc=wacc, u=par.BATTERY_LIFETIME_BATTERY_PACK)
    )
    ep_battery_power = (
        economics.annuity(capex=p["battery_capex_power"],
                          n=par.LIFETIME, wacc=wacc, u=par.BATTERY_LIFETIME_SYSTEM)
        + par.BATTERY_OPEX_FIX
    )
    ep_saltcavern_energy = (
        economics.annuity(capex=p["saltcavern_capex"],
                          n=par.SALTCAVERN_LIFETIME, wacc=wacc)
        + economics.annuity(capex=p["saltcavern_capex_cushion_gas"],
                            n=par.SALTCAVERN_LIFETIME, wacc=wacc)
        + par.SALTCAVERN_OPEX
    )
    ep_saltcavern_power = economics.annuity(
        capex=p["saltcavern_capex_compressor"],
        n=par.SALTCAVERN_LIFETIME, wacc=wacc,
        u=par.SALTCAVERN_LIFETIME_COMPRESSOR,
    )
    ep_thermalstorage_energy = (
        economics.annuity(capex=par.THERMALSTORAGE_CAPEX,
                          n=par.THERMALSTORAGE_LIFETIME, wacc=wacc)
        + par.THERMALSTORAGE_OPEX
    )

    # ── Build energy system ───────────────────────────────────────────────────
    energysystem = solph.EnergySystem(
        timeindex=timeseries.index, infer_last_interval=True)

    bus_electricity = solph.Bus(label='electricity')
    bus_H2          = solph.Bus(label='hydrogen')
    bus_heat        = solph.Bus(label='heat')

    pv = solph.components.Source(
        label="pv",
        outputs={bus_electricity: solph.Flow(
            fix=timeseries["PV"],
            nominal_value=solph.Investment(ep_costs=ep_pv, minimum=0, existing=0),
            variable_costs=0,
        )},
    )

    wind = solph.components.Source(
        label="wind",
        outputs={bus_electricity: solph.Flow(
            fix=timeseries["Wind"],
            nominal_value=solph.Investment(ep_costs=ep_wind, existing=0),
            variable_costs=0,
        )},
    )

    electrolyser = solph.components.Converter(
        label="electrolyser",
        inputs={bus_electricity: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=ep_electrolyser, minimum=0, existing=0),
            min=par.ELECTROLYSER_MIN_LOAD,
            variable_costs=0,
        )},
        outputs={bus_H2: solph.Flow(), bus_heat: solph.Flow()},
        conversion_factors={
            bus_H2:   par.ELECTROLYSER_EFFICIENCY,
            bus_heat: par.ELECTROLYSER_RECOVERABLE_HEAT,
        },
    )

    heatpump = solph.components.Converter(
        label="heatpump",
        inputs={bus_electricity: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=ep_heatpump, minimum=0, existing=0),
            min=par.HEATPUMP_MIN_LOAD,
            variable_costs=par.HEATPUMP_OPEX_VAR,
        )},
        outputs={bus_heat: solph.Flow()},
        conversion_factors={bus_heat: list(cop_ts)},
    )

    fuelcell = solph.components.Converter(
        label="fuelcell",
        inputs={bus_H2: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=ep_fuelcell, minimum=0, existing=0),
            min=par.FUELCELL_MIN_LOAD,
            variable_costs=0,
        )},
        outputs={bus_heat: solph.Flow(), bus_electricity: solph.Flow()},
        conversion_factors={
            bus_heat:        par.FUELCELL_EFFICIENCY_HEAT,
            bus_electricity: par.FUELCELL_EFFICIENCY_ELECTRICITY,
        },
    )

    battery = solph.components.GenericStorage(
        label="battery",
        inputs={bus_electricity: solph.Flow(
            nominal_value=solph.Investment(ep_costs=ep_battery_power),
            variable_costs=par.BATTERY_OPEX_VAR / 2,
        )},
        outputs={bus_electricity: solph.Flow(
            nominal_value=solph.Investment(ep_costs=0),
            variable_costs=par.BATTERY_OPEX_VAR / 2,
        )},
        nominal_storage_capacity=solph.Investment(
            ep_costs=ep_battery_energy, minimum=0, existing=0),
        invest_relation_input_capacity=1 / par.BATTERY_MIN_CHARGE_TIME,
        invest_relation_input_output=1,
        inflow_conversion_factor=par.BATTERY_EFFICIENCY_CHARGE,
        outflow_conversion_factor=par.BATTERY_EFFICIENCY_DISCHARGE,
        loss_rate=par.BATTERY_SELF_DISCHARGE_RATE,
        initial_storage_level=0,
    )

    saltcavern = solph.components.GenericStorage(
        label="saltcavern",
        inputs={bus_H2: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=ep_saltcavern_power, minimum=0, existing=0),
            variable_costs=par.SALTCAVERN_OPEX_COMPRESSOR,
        )},
        outputs={bus_H2: solph.Flow(
            nominal_value=solph.Investment(ep_costs=0))},
        nominal_storage_capacity=solph.Investment(
            ep_costs=ep_saltcavern_energy, minimum=0, existing=0),
        invest_relation_input_output=1,
        inflow_conversion_factor=1,
        outflow_conversion_factor=par.SALTCAVERN_EFFICIENCY,
        loss_rate=par.SALTCAVERN_SELF_DISCHARGE_RATE,
        min_storage_level=par.SALTCAVERN_CUSHIONGAS_FRACTION,
        initial_storage_level=par.SALTCAVERN_CUSHIONGAS_FRACTION,
    )

    thermalstorage = solph.components.GenericStorage(
        label="thermal_storage",
        inputs={bus_heat: solph.Flow(
            nominal_value=solph.Investment(ep_costs=0), variable_costs=0)},
        outputs={bus_heat: solph.Flow(
            nominal_value=solph.Investment(ep_costs=0), variable_costs=0)},
        nominal_storage_capacity=solph.Investment(
            ep_costs=ep_thermalstorage_energy, minimum=0, existing=0),
        invest_relation_input_capacity=1 / par.THERMALSTORAGE_MIN_CHARGE_TIME,
        invest_relation_output_capacity=1 / par.THERMALSTORAGE_MIN_CHARGE_TIME,
        inflow_conversion_factor=par.THERMALSTORAGE_EFFICIENCY_CHARGE,
        outflow_conversion_factor=par.THERMALSTORAGE_EFFICIENCY_DISCHARGE,
        loss_rate=par.THERMALSTORAGE_SELF_DISCHARGE_RATE,
        min_storage_level=0,
        initial_storage_level=0,
    )

    ElectricityDemand = solph.components.Sink(
        label="Electricity_demand",
        inputs={bus_electricity: solph.Flow(
            fix=timeseries["Electricity_Demand"], nominal_value=1, variable_costs=0)},
    )
    H2Demand = solph.components.Sink(
        label="H2_demand",
        inputs={bus_H2: solph.Flow(
            fix=timeseries["H2_Demand"], nominal_value=1, variable_costs=0)},
    )
    HeatDemand = solph.components.Sink(
        label="Heat_demand",
        inputs={bus_heat: solph.Flow(
            fix=timeseries["Heat_Demand"], nominal_value=1, variable_costs=0)},
    )
    electricity_curtailment = solph.components.Sink(
        label="electricity_curtailment",
        inputs={bus_electricity: solph.Flow(variable_costs=0)},
    )
    H2_curtailment = solph.components.Sink(
        label="H2_curtailment",
        inputs={bus_H2: solph.Flow(variable_costs=0)},
    )
    heat_curtailment = solph.components.Sink(
        label="heat_curtailment",
        inputs={bus_heat: solph.Flow(variable_costs=0)},
    )

    energysystem.add(
        bus_electricity, bus_H2, bus_heat,
        pv, wind,
        electrolyser, fuelcell, heatpump,
        battery, saltcavern, thermalstorage,
        ElectricityDemand, H2Demand, HeatDemand,
        electricity_curtailment, H2_curtailment, heat_curtailment,
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    model = solph.Model(energysystem)
    model.solve(solver='cbc', solve_kwargs={'tee': False})

    results = solph.processing.results(model)

    # ── Extract capacities ────────────────────────────────────────────────────
    cap_pv             = results[(pv, bus_electricity)]['scalars']['invest']
    cap_wind           = results[(wind, bus_electricity)]['scalars']['invest']
    cap_electrolyser   = results[(bus_electricity, electrolyser)]['scalars']['invest']
    cap_fuelcell       = results[(bus_H2, fuelcell)]['scalars']['invest']
    cap_heatpump       = results[(bus_electricity, heatpump)]['scalars']['invest']
    cap_battery        = results[(battery, None)]['scalars']['invest']
    cap_saltcavern     = results[(saltcavern, None)]['scalars']['invest']
    cap_thermalstorage = results[(thermalstorage, None)]['scalars']['invest']
    cap_battery_power  = results[(bus_electricity, battery)]['scalars']['invest']
    cap_saltcavern_power = results[(bus_H2, saltcavern)]['scalars']['invest']

    # ── Extract key timeseries for cost attribution ───────────────────────────
    el_input         = results[(bus_electricity, electrolyser)]['sequences']['flow'].iloc[:-1]
    el_output        = results[(electrolyser, bus_H2)]['sequences']['flow'].iloc[:-1]
    el_output_heat   = results[(electrolyser, bus_heat)]['sequences']['flow'].iloc[:-1]
    fc_input         = results[(bus_H2, fuelcell)]['sequences']['flow'].iloc[:-1]
    fc_output_el     = results[(fuelcell, bus_electricity)]['sequences']['flow'].iloc[:-1]
    fc_output_heat   = results[(fuelcell, bus_heat)]['sequences']['flow'].iloc[:-1]
    hp_input         = results[(bus_electricity, heatpump)]['sequences']['flow'].iloc[:-1]
    bat_charge       = results[(bus_electricity, battery)]['sequences']['flow'].iloc[:-1]
    bat_discharge    = results[(battery, bus_electricity)]['sequences']['flow'].iloc[:-1]
    sc_charge        = results[(bus_H2, saltcavern)]['sequences']['flow'].iloc[:-1]
    sc_discharge     = results[(saltcavern, bus_H2)]['sequences']['flow'].iloc[:-1]
    ts_discharge     = results[(thermalstorage, bus_heat)]['sequences']['flow'].iloc[:-1]

    el_demand_ts   = results[(bus_electricity, ElectricityDemand)]['sequences']['flow'].iloc[:-1]
    h2_demand_ts   = results[(bus_H2, H2Demand)]['sequences']['flow'].iloc[:-1]
    heat_demand_ts = results[(bus_heat, HeatDemand)]['sequences']['flow'].iloc[:-1]

    # ── Cost attribution (mirrors main model exactly) ─────────────────────────
    total_cost = model.objective()

    total_h2_mwh   = h2_demand_ts.sum()
    total_el_mwh   = el_demand_ts.sum()
    total_heat_mwh = heat_demand_ts.sum()
    total_energy   = total_h2_mwh + total_el_mwh + total_heat_mwh

    ep_map = {
        "pv":                ep_pv,
        "wind":              ep_wind,
        "electrolyser":      ep_electrolyser,
        "heatpump":          ep_heatpump,
        "fuelcell":          ep_fuelcell,
        "battery_energy":    ep_battery_energy,
        "battery_power":     ep_battery_power,
        "saltcavern_energy": ep_saltcavern_energy,
        "saltcavern_power":  ep_saltcavern_power,
        "thermalstorage_energy": ep_thermalstorage_energy,
        "thermalstorage_power":  0.0,
    }
    inv = {
        "pv": cap_pv, "wind": cap_wind,
        "electrolyser": cap_electrolyser, "heatpump": cap_heatpump,
        "fuelcell": cap_fuelcell,
        "battery_energy": cap_battery, "battery_power": cap_battery_power,
        "saltcavern_energy": cap_saltcavern, "saltcavern_power": cap_saltcavern_power,
        "thermalstorage_energy": cap_thermalstorage, "thermalstorage_power": 0.0,
    }

    cost_pv           = ep_map["pv"]              * inv["pv"]
    cost_wind         = ep_map["wind"]            * inv["wind"]
    cost_electrolyser = ep_map["electrolyser"]    * inv["electrolyser"]
    cost_heatpump     = ep_map["heatpump"]        * inv["heatpump"]
    cost_fuelcell     = ep_map["fuelcell"]        * inv["fuelcell"]
    cost_battery      = (ep_map["battery_energy"] * inv["battery_energy"]
                       + ep_map["battery_power"]  * inv["battery_power"])
    cost_saltcavern   = (ep_map["saltcavern_energy"] * inv["saltcavern_energy"]
                       + ep_map["saltcavern_power"]  * inv["saltcavern_power"])
    cost_thermstor    = ep_map["thermalstorage_energy"] * inv["thermalstorage_energy"]

    var_hp_opex  = (hp_input  * par.HEATPUMP_OPEX_VAR).sum()
    var_bat_opex = ((bat_charge + bat_discharge) * par.BATTERY_OPEX_VAR / 2).sum()
    var_sc_opex  = (sc_charge * par.SALTCAVERN_OPEX_COMPRESSOR).sum()

    # Electrolyser cost split
    el_tot_out = el_output.sum() + el_output_heat.sum()
    f_h2   = el_output.sum()      / el_tot_out if el_tot_out > 0 else 1.0
    f_heat = el_output_heat.sum() / el_tot_out if el_tot_out > 0 else 0.0
    cost_el_h2   = cost_electrolyser * f_h2
    cost_el_heat = cost_electrolyser * f_heat

    # Fuel cell cost split
    fc_tot_out = fc_output_el.sum() + fc_output_heat.sum()
    f_fc_el   = fc_output_el.sum()   / fc_tot_out if fc_tot_out > 0 else 0.5
    f_fc_heat = fc_output_heat.sum() / fc_tot_out if fc_tot_out > 0 else 0.5
    cost_fc_el   = cost_fuelcell * f_fc_el
    cost_fc_heat = cost_fuelcell * f_fc_heat

    # Supply allocation
    cost_supply = cost_pv + cost_wind
    frac_h2   = total_h2_mwh   / total_energy if total_energy > 0 else 0.0
    frac_el   = total_el_mwh   / total_energy if total_energy > 0 else 0.0
    frac_heat = total_heat_mwh / total_energy if total_energy > 0 else 0.0

    cost_h2_total   = (cost_supply * frac_h2 + cost_el_h2
                       + cost_saltcavern + var_sc_opex)
    cost_el_total   = (cost_supply * frac_el  + cost_battery
                       + var_bat_opex + cost_fc_el)
    cost_heat_total = (cost_supply * frac_heat + cost_heatpump + var_hp_opex
                       + cost_thermstor + cost_el_heat + cost_fc_heat)

    lcoh2 = cost_h2_total   / total_h2_mwh   if total_h2_mwh   > 0 else np.nan
    lcoel = cost_el_total   / total_el_mwh   if total_el_mwh   > 0 else np.nan
    lcoh  = cost_heat_total / total_heat_mwh if total_heat_mwh > 0 else np.nan
    lcoe  = total_cost      / total_energy   if total_energy    > 0 else np.nan

    lcoh2_kg = lcoh2 * par.H2_CALORIFIC_VALUE_LHV if not np.isnan(lcoh2) else np.nan

    return {
        # sampled parameters
        **{f"param_{k}": v for k, v in p.items()},
        # capacities
        "cap_pv_mw":             cap_pv,
        "cap_wind_mw":           cap_wind,
        "cap_res_total_mw":      cap_pv + cap_wind,
        "cap_electrolyser_mw":   cap_electrolyser,
        "cap_fuelcell_mw":       cap_fuelcell,
        "cap_heatpump_mw":       cap_heatpump,
        "cap_battery_mwh":       cap_battery,
        "cap_saltcavern_mwh":    cap_saltcavern,
        "cap_thermalstorage_mwh":cap_thermalstorage,
        # costs
        "total_system_cost_eur": total_cost,
        "lcoh2_eur_per_mwh":     lcoh2,
        "lcoh2_eur_per_kg":      lcoh2_kg,
        "lcoel_eur_per_mwh":     lcoel,
        "lcoh_eur_per_mwh":      lcoh,
        "lcoe_eur_per_mwh":      lcoe,
    }


# =============================================================================
# MONTE CARLO LOOP
# =============================================================================

records   = []
failed    = 0

for i in tqdm(range(N_SAMPLES), desc="Monte Carlo"):
    sample = sample_parameters()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_single(sample)
        if result is not None:
            result["sample_id"] = i
            records.append(result)
    except Exception:
        failed += 1
        print(f"\n[WARNING] Sample {i} failed:")
        traceback.print_exc()

print(f"\nCompleted: {len(records)} successful, {failed} failed out of {N_SAMPLES}.")

df = pd.DataFrame(records)
df.to_csv(RESULTS_CSV, index=False)
print(f"Results saved to: {RESULTS_CSV}")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

metrics = ["lcoh2_eur_per_mwh", "lcoh2_eur_per_kg",
           "lcoel_eur_per_mwh", "lcoh_eur_per_mwh", "lcoe_eur_per_mwh"]

print("\n" + "=" * 70)
print("  MONTE CARLO — SUMMARY STATISTICS")
print("=" * 70)
for m in metrics:
    col = df[m].dropna()
    print(f"  {m:<30}  mean={col.mean():>8.2f}  std={col.std():>7.2f}"
          f"  p5={col.quantile(0.05):>8.2f}  p95={col.quantile(0.95):>8.2f}")
print("=" * 70)

cap_cols = ["cap_pv_mw", "cap_wind_mw", "cap_electrolyser_mw",
            "cap_fuelcell_mw", "cap_heatpump_mw",
            "cap_battery_mwh", "cap_saltcavern_mwh"]
print("\n  OPTIMAL CAPACITY STATISTICS")
print("=" * 70)
for c in cap_cols:
    col = df[c].dropna()
    print(f"  {c:<30}  mean={col.mean():>8.1f}  std={col.std():>7.1f}"
          f"  p5={col.quantile(0.05):>8.1f}  p95={col.quantile(0.95):>8.1f}")
print("=" * 70)


# =============================================================================
# PLOTS
# =============================================================================

# ── 1. Histogram grid for all four levelised costs ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

plot_specs = [
    ("lcoh2_eur_per_mwh", r"LCOH$_2$ [€/MWh]",   "tab:green"),
    ("lcoel_eur_per_mwh", r"LCOE$_l$ [€/MWh]",    "tab:olive"),
    ("lcoh_eur_per_mwh",  r"LCOH [€/MWh]",         "tab:orange"),
    ("lcoe_eur_per_mwh",  r"LCOE system [€/MWh]",  "slateblue"),
]

for ax, (col, label, color) in zip(axes.flat, plot_specs):
    data = df[col].dropna()
    ax.hist(data, bins=30, color=color, alpha=0.80, edgecolor="none")
    ax.axvline(data.mean(),          color="black", lw=1.5, ls="--",
               label=f"Mean: {data.mean():.1f}")
    ax.axvline(data.quantile(0.05),  color="black", lw=1.0, ls=":",
               label=f"P5:   {data.quantile(0.05):.1f}")
    ax.axvline(data.quantile(0.95),  color="black", lw=1.0, ls=":",
               label=f"P95:  {data.quantile(0.95):.1f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.suptitle(f"Monte Carlo Levelised Cost Distributions  (N={len(df)})", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f"{SCENARIO_NAME}_mc_lc_histograms.pdf",
            bbox_inches="tight")
plt.show()


# ── 2. Capacity box plots ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

cap_labels = {
    "cap_pv_mw":           "PV [MW]",
    "cap_wind_mw":         "Wind [MW]",
    "cap_electrolyser_mw": "Electrolyser [MW]",
    "cap_fuelcell_mw":     "Fuel Cell [MW]",
    "cap_heatpump_mw":     "Heat Pump [MW]",
    "cap_battery_mwh":     "Battery [MWh]",
    "cap_saltcavern_mwh":  "Salt Cavern [MWh]",
}

cap_data  = [df[k].dropna().values for k in cap_labels]
cap_names = list(cap_labels.values())

bp = ax.boxplot(cap_data, patch_artist=True, medianprops=dict(color="black", lw=1.5))
colors_box = ["peru", "skyblue", "tab:blue", "gold", "tab:red", "purple", "brown"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticklabels(cap_names, rotation=30, ha="right")
ax.set_ylabel("Optimal capacity")
ax.set_title(f"Optimal Capacity Distributions — Monte Carlo (N={len(df)})")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f"{SCENARIO_NAME}_mc_capacity_boxplots.pdf",
            bbox_inches="tight")
plt.show()


# ── 3. Tornado plot — sensitivity of LCOH2 to each parameter ─────────────────
# Uses Spearman rank correlation between each sampled parameter and LCOH2.

from scipy.stats import spearmanr

param_cols = [c for c in df.columns if c.startswith("param_")]
target     = df["lcoh2_eur_per_mwh"].dropna()
idx        = target.index

correlations = {}
for col in param_cols:
    x = df.loc[idx, col].dropna()
    common = x.index.intersection(target.index)
    if len(common) < 5:
        continue
    rho, _ = spearmanr(x.loc[common], target.loc[common])
    correlations[col.replace("param_", "")] = rho

corr_series = pd.Series(correlations).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
colors_tornado = ["firebrick" if v > 0 else "steelblue" for v in corr_series.values]
ax.barh(corr_series.index, corr_series.values, color=colors_tornado, alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Spearman rank correlation with LCOH₂")
ax.set_title(f"Parameter Sensitivity — Tornado Plot\n"
             f"Correlation with LCOH₂  (N={len(df)})")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f"{SCENARIO_NAME}_mc_tornado_lcoh2.pdf",
            bbox_inches="tight")
plt.show()


# ── 4. LCOH2 vs WACC scatter ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(df["param_wacc"] * 100, df["lcoh2_eur_per_mwh"],
                c=df["cap_electrolyser_mw"], cmap="viridis",
                s=15, alpha=0.7)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Electrolyser capacity [MW]")
ax.set_xlabel("WACC [%]")
ax.set_ylabel(r"LCOH$_2$ [€/MWh]")
ax.set_title(r"LCOH$_2$ vs WACC, coloured by electrolyser capacity")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f"{SCENARIO_NAME}_mc_lcoh2_vs_wacc.pdf",
            bbox_inches="tight")
plt.show()


print("\nAll MC plots saved.")
print("Done.")
