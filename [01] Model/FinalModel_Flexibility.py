'''Author: Gonçalo Costa Pina
Date_Created: 2026-01-30 (30th January 2026)
Date_Modified: 2026-05-21 (21st May 2026)

----------------------------------------------

Defines the main working script for the project, which:
- Sets up and solves the oemof optimization model
- Produces visualisations of the main results
storage_utilisation_rate
Important notes:
- This script is intended to be run from the [01] Model folder, and imports functions from funcs.py.
- The variable "SCENARIO_NAME" is highly relevant, as it determines the where results are saved, according to the scenario that is being analysed.
'''



import pandas as pd
from pathlib import Path
import numpy as np
from oemof import solph
from oemof.solph import processing, views
from oemof.tools import economics
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyomo.environ as po
from Parameters import parameters as par
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import numpy_financial as npf   # pip install numpy-financial
from collections import defaultdict
import funcs as func
import flexibility_metrics as fm
import networkx as nx
from oemof.network.graph import create_nx_graph
from pyvis.network import Network
import plotly.graph_objects as go
from playwright.sync_api import sync_playwright

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
dt = 1  # hours per timestep
CURTAILMENT_FRACTION = par.CURTAILMENT_SHARE 
print(f"Assuming a curtailment share of {CURTAILMENT_FRACTION*100:.1f}% for the renewable generation (PV + Wind) as a starting point for flexibility needs assessment.")

# =============================================================================
# SCENARIO DEFINITION
# =============================================================================
SCENARIO_NAME = "FINAL_MODEL_FIXEDCAPACITIES_FIXEDCURTAILMENT_NEW"   # ← change per run


# ============================================================================
# IMPORTING PROFILES
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent
data_path = (DATA_DIR.parent / "[02] Data" / "Supply")

pv_file   = data_path / "ninja_pv_51.9244_4.4778_corrected.csv"
wind_file = data_path / "ninja_wind_51.9244_4.4778_corrected.csv"

df_pv = pd.read_csv(pv_file, sep=",", skiprows=3)
df_pv = df_pv[["time", "electricity"]].copy()
df_pv["electricity"] = pd.to_numeric(df_pv["electricity"], errors="coerce")
df_pv = df_pv.dropna(subset=["electricity"])
df_pv["time"] = pd.to_datetime(df_pv["time"], format="%Y-%m-%d %H:%M")
df_pv = df_pv.sort_values("time")
print("PV profile loaded:", len(df_pv), "rows")

df_wind = pd.read_csv(wind_file, sep=",", skiprows=3)
df_wind = df_wind[["time", "electricity"]].copy()
df_wind["electricity"] = pd.to_numeric(df_wind["electricity"], errors="coerce")
df_wind = df_wind.dropna(subset=["electricity"])
df_wind["time"] = pd.to_datetime(df_wind["time"], format="%Y-%m-%d %H:%M")
df_wind = df_wind.sort_values("time")
print("Wind profile loaded:", len(df_wind), "rows")

# ── Demand Data Frames (H2, Electricity, Heat) ────────────────────────────────
demand_file = (
    DATA_DIR.parent / "[02] Data" / "Demand" / "Demand_Profiles.csv"
)

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

demand_h2   = df_dem["Rotterdam_total_gas_demand [MW]"].values
demand_el   = df_dem["Rotterdam_electricity_load [MW]"].values
demand_heat = df_dem["Rotterdam_total_heat_demand [MW]"].values
print("Demand profiles loaded:", len(df_dem), "rows")

timeseries = pd.DataFrame({
    "PV":                 df_pv["electricity"].values,
    "Wind":               df_wind["electricity"].values,
    "H2_Demand":          demand_h2,
    "Electricity_Demand": demand_el,
    "Heat_Demand":        demand_heat,
}, index=df_dem["Datetime (UTC)"])

timeseries.index = pd.DatetimeIndex(timeseries.index).round("h")
timeseries.index.name = "time"
timeseries = timeseries.asfreq("h")


# =============================================================================
# OEMOF MODEL SETUP
# =============================================================================

energysystem = solph.EnergySystem(timeindex=timeseries.index, infer_last_interval=True)

bus_electricity = solph.Bus(label='electricity')
bus_H2          = solph.Bus(label='hydrogen')
bus_heat        = solph.Bus(label='heat')
#FIXME: Supply and PtX capacities are fixed to analyse how storage changes with curtailment
#TODO: Make minimum = 0 and remove maximum
# ── PV ────────────────────────────────────────────────────────────────────────
pv = solph.components.Source(
    label="pv",
    outputs={bus_electricity: solph.Flow(
        fix=timeseries["PV"],
        nominal_value=solph.Investment(
            ep_costs=(
                economics.annuity(capex=par.PV_CAPEX_PANELS,   n=par.PV_LIFETIME_SYSTEM, wacc=par.WACC)
                + economics.annuity(capex=par.PV_CAPEX_INVERTER, n=par.PV_LIFETIME_SYSTEM, wacc=par.WACC, u=par.PV_LIFETIME_INVERTER)
                + par.PV_OPEX
            ),
            minimum= 2127.3, 
            maximum = 2127.3,
            existing=0,
        ),
        variable_costs=0,
    )},
)

# ── Wind ──────────────────────────────────────────────────────────────────────
wind = solph.components.Source(
    label="wind",
    outputs={bus_electricity: solph.Flow(
        fix=timeseries["Wind"],
        nominal_value=solph.Investment(
            ep_costs=economics.annuity(capex=par.WIND_CAPEX, n=par.WIND_LIFETIME_SYSTEM, wacc=par.WACC) + par.WIND_OPEX,
            minimum= 2534.7, 
            maximum = 2534.7,
            existing=0,
        ),
        variable_costs=0,
    )},
)

# ── Electrolyser ──────────────────────────────────────────────────────────────
electrolyser = solph.components.Converter(
    label="electrolyser",
    inputs={bus_electricity: solph.Flow(
        nominal_value=solph.Investment(
            ep_costs=(
                economics.annuity(capex=par.ELECTROLYSER_CAPEX_SYSTEM, n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC)
                + economics.annuity(capex=par.ELECTROLYSER_CAPEX_STACK,  n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC, u=par.ELECTROLYSER_LIFETIME_STACK)
                + par.ELECTROLYSER_OPEX
            ),
            minimum= 681.1, 
            maximum = 681.1, 
            existing=0,
        ),
        min=par.ELECTROLYSER_MIN_LOAD,
        variable_costs=0,
    )},
    outputs={bus_H2: solph.Flow(), bus_heat: solph.Flow()},
    conversion_factors={bus_H2: par.ELECTROLYSER_EFFICIENCY, bus_heat: par.ELECTROLYSER_RECOVERABLE_HEAT},
)

# ── Heat Pump ─────────────────────────────────────────────────────────────────
months = timeseries.index.month
cop_ts = np.where(
    (months >= par.HEATPUMP_SUMMERTIME_START) & (months <= par.HEATPUMP_SUMMERTIME_END),
    par.HEATPUMP_COP * par.HEATPUMP_COP_SUMMERFACTOR,
    par.HEATPUMP_COP * par.HEATPUMP_COP_WINTERFACTOR,
)
cop_ts_list = list(cop_ts)

heatpump = solph.components.Converter(
    label="heatpump",
    inputs={bus_electricity: solph.Flow(
        nominal_value=solph.Investment(
            ep_costs=(
                economics.annuity(capex=par.HEATPUMP_CAPEX, n=par.HEATPUMP_LIFETIME, wacc=par.WACC)
                + par.HEATPUMP_OPEX_FIX
            ),
            minimum=199.6, 
            maximum = 199.6,
            existing=0,
        ),
        min=par.HEATPUMP_MIN_LOAD,
        variable_costs=par.HEATPUMP_OPEX_VAR,
    )},
    outputs={bus_heat: solph.Flow()},
    conversion_factors={bus_heat: cop_ts_list},
)

# ── CHP (fuel cell) ───────────────────────────────────────────────────────────
fuelcell = solph.components.Converter(
    label="fuelcell",
    inputs={bus_H2: solph.Flow(
        nominal_value=solph.Investment(
            ep_costs=(
                economics.annuity(capex=par.FUELCELL_CAPEX_SYSTEM, n=par.FUELCELL_LIFETIME_SYSTEM, wacc=par.WACC)
                + economics.annuity(capex=par.FUELCELL_CAPEX_STACK,  n=par.FUELCELL_LIFETIME_SYSTEM, wacc=par.WACC, u=par.FUELCELL_LIFETIME_STACK)
                + par.FUELCELL_OPEX_FIX
            ),
            minimum=10.3, 
            maximum = 10.3,
            existing=0,
        ),
        min=par.FUELCELL_MIN_LOAD,
        variable_costs=0,
    )},
    outputs={bus_heat: solph.Flow(), bus_electricity: solph.Flow()},
    conversion_factors={bus_heat: par.FUELCELL_EFFICIENCY_HEAT, bus_electricity: par.FUELCELL_EFFICIENCY_ELECTRICITY},
)

# ── Battery ───────────────────────────────────────────────────────────────────
battery = solph.components.GenericStorage( #TODO: heat generation 
    label="battery",
    inputs={
        bus_electricity: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=(
                    economics.annuity(capex=par.BATTERY_CAPEX_POWER, n=par.LIFETIME, wacc=par.WACC, u=par.BATTERY_LIFETIME_SYSTEM)
                    + par.BATTERY_OPEX_FIX
                ),
            ),
            variable_costs=par.BATTERY_OPEX_VAR / 2,
        )
    },
    outputs={
        bus_electricity: solph.Flow(
            nominal_value=solph.Investment(ep_costs=0),
            variable_costs=par.BATTERY_OPEX_VAR / 2,
        )
    },
    nominal_storage_capacity=solph.Investment(
        ep_costs=(
            economics.annuity(capex=par.BATTERY_CAPEX_SYSTEM,       n=par.LIFETIME, wacc=par.WACC, u=par.BATTERY_LIFETIME_SYSTEM)
            + economics.annuity(capex=par.BATTERY_CAPEX_BATTERY_PACK, n=par.LIFETIME, wacc=par.WACC, u=par.BATTERY_LIFETIME_BATTERY_PACK)
        ),
        minimum=0, existing=0,
    ),
    invest_relation_input_capacity=1 / par.BATTERY_MIN_CHARGE_TIME,
    invest_relation_input_output=1,
    inflow_conversion_factor=par.BATTERY_EFFICIENCY_CHARGE,
    outflow_conversion_factor=par.BATTERY_EFFICIENCY_DISCHARGE,
    loss_rate=par.BATTERY_SELF_DISCHARGE_RATE,
    initial_storage_level=0,
)

# ── Salt Cavern ───────────────────────────────────────────────────────────────
saltcavern = solph.components.GenericStorage( # TODO: shrinkage rate; limit poewr flows
    label="saltcavern",
    inputs={
        bus_H2: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=economics.annuity(
                    capex=par.SALTCAVERN_CAPEX_COMPRESSOR,
                    n=par.SALTCAVERN_LIFETIME, wacc=par.WACC,
                    u=par.SALTCAVERN_LIFETIME_COMPRESSOR,
                ),
                minimum=0, existing=0,
            ),
            variable_costs=par.SALTCAVERN_OPEX_COMPRESSOR,
        )
    },
    outputs={
        bus_H2: solph.Flow(nominal_value=solph.Investment(ep_costs=0))
    },
    nominal_storage_capacity=solph.Investment(
        ep_costs=(
            economics.annuity(capex=par.SALTCAVERN_CAPEX,             n=par.SALTCAVERN_LIFETIME, wacc=par.WACC)
            + economics.annuity(capex=par.SALTCAVERN_CAPEX_CUSHION_GAS, n=par.SALTCAVERN_LIFETIME, wacc=par.WACC)
            + par.SALTCAVERN_OPEX
        ),
        minimum=0, existing=0,
    ),
    invest_relation_input_output=1,
    inflow_conversion_factor=1,
    outflow_conversion_factor=par.SALTCAVERN_EFFICIENCY,
    loss_rate=par.SALTCAVERN_SELF_DISCHARGE_RATE,
    min_storage_level=par.SALTCAVERN_CUSHIONGAS_FRACTION,
    initial_storage_level=par.SALTCAVERN_CUSHIONGAS_FRACTION,
)

# ── Thermal Storage ───────────────────────────────────────────────────────────
thermalstorage = solph.components.GenericStorage(
    label="thermal_storage",
    inputs={
        bus_heat: solph.Flow(
            nominal_value=solph.Investment(ep_costs=0),
            variable_costs=0,
        )
    },
    outputs={
        bus_heat: solph.Flow(
            nominal_value=solph.Investment(ep_costs=0),
            variable_costs=0,
        )
    },
    nominal_storage_capacity=solph.Investment(
        ep_costs=(
            economics.annuity(capex=par.THERMALSTORAGE_CAPEX, n=par.THERMALSTORAGE_LIFETIME, wacc=par.WACC)
            + par.THERMALSTORAGE_OPEX
        ),
        minimum=0, existing=0,
    ),
    invest_relation_input_capacity=1 / par.THERMALSTORAGE_MIN_CHARGE_TIME,
    invest_relation_output_capacity=1 / par.THERMALSTORAGE_MIN_CHARGE_TIME,
    inflow_conversion_factor=par.THERMALSTORAGE_EFFICIENCY_CHARGE,
    outflow_conversion_factor=par.THERMALSTORAGE_EFFICIENCY_DISCHARGE,
    loss_rate=par.THERMALSTORAGE_SELF_DISCHARGE_RATE,
    min_storage_level=0,
    initial_storage_level=0,
)

# ── Demand Sinks ──────────────────────────────────────────────────────────────
ElectricityDemand = solph.components.Sink(
    label="Electricity_demand",
    inputs={bus_electricity: solph.Flow(fix=timeseries["Electricity_Demand"], nominal_value=1, variable_costs=0)},
)
H2Demand = solph.components.Sink(
    label="H2_demand",
    inputs={bus_H2: solph.Flow(fix=timeseries["H2_Demand"], nominal_value=1, variable_costs=0)},
)
HeatDemand = solph.components.Sink(
    label="Heat_demand",
    inputs={bus_heat: solph.Flow(fix=timeseries["Heat_Demand"], nominal_value=1, variable_costs=0)},
)

# ── Curtailment Sinks ─────────────────────────────────────────────────────────
electricity_curtailement = solph.components.Sink(
    label="electricity_curtailement",
    inputs={bus_electricity: solph.Flow(variable_costs=0)},
)
H2_curtailement = solph.components.Sink(
    label="H2_curtailement",
    inputs={bus_H2: solph.Flow(variable_costs=0)},
)
heat_curtailement = solph.components.Sink(
    label="heat_curtailement",
    inputs={bus_heat: solph.Flow(variable_costs=0)},
)

print("Max Electricity demand (MWh/h):", timeseries["Electricity_Demand"].max())
print("Annual Electricity demand (MWh):", timeseries["Electricity_Demand"].sum())
print("Max H2 demand (MWh/h):",          timeseries["H2_Demand"].max())
print("Annual H2 demand (MWh):",          timeseries["H2_Demand"].sum())
print("Max Heat demand (MWh/h):",         timeseries["Heat_Demand"].max())
print("Annual Heat demand (MWh):",         timeseries["Heat_Demand"].sum())

energysystem.add(
    bus_electricity, bus_H2, bus_heat,
    pv, wind,
    electrolyser, fuelcell, heatpump,
    battery, saltcavern, thermalstorage,
    ElectricityDemand, H2Demand, HeatDemand,
    electricity_curtailement, H2_curtailement, heat_curtailement,
)


# =============================================================================
# OEMOF MODEL OPTIMIZATION AND RESULTS EXTRACTION
# =============================================================================

model = solph.Model(energysystem)

model = func.add_curtailment_limit(
    model                    = model,
    bus_electricity          = bus_electricity,
    bus_H2                   = bus_H2,
    bus_heat                 = bus_heat,
    electricity_curtailement = electricity_curtailement,
    H2_curtailement          = H2_curtailement,
    heat_curtailement        = heat_curtailement,
    pv                       = pv,
    wind                     = wind,
    fraction                 = CURTAILMENT_FRACTION,
)


print("Running optimization...\n")
model.solve(solver='cbc', solve_kwargs={'tee': True})
print("Processing results...\n")


results        = solph.processing.results(model)
string_results = views.convert_keys_to_strings(results)

# ── Optimal capacities ────────────────────────────────────────────────────────
capacity_pv             = results[(pv, bus_electricity)]['scalars']['invest']
capacity_wind           = results[(wind, bus_electricity)]['scalars']['invest']
capacity_electrolyser   = results[(bus_electricity, electrolyser)]['scalars']['invest']
capacity_fuelcell       = results[(bus_H2, fuelcell)]['scalars']['invest']
capacity_heatpump       = results[(bus_electricity, heatpump)]['scalars']['invest']
capacity_battery        = results[(battery, None)]['scalars']['invest']
capacity_saltcavern     = results[(saltcavern, None)]['scalars']['invest']
capacity_thermalstorage = results[(thermalstorage, None)]['scalars']['invest']

print("="*90)
print("🎯 OPTIMISATION RESULTS")
print("="*90)
print(f"  PV:                      {capacity_pv:.1f} MW")
print(f"  Wind:                    {capacity_wind:.1f} MW")
print(f"  TOTAL RES:               {capacity_pv + capacity_wind:.1f} MW")
print(f"  Electrolyser:            {capacity_electrolyser:.1f} MW")
print(f"  fuelcell:                {capacity_fuelcell:.1f} MW")
print(f"  Heat Pump:               {capacity_heatpump:.1f} MW")
print(f"  Battery:                 {capacity_battery:.1f} MWh")
print(f"  Salt Cavern:             {capacity_saltcavern:.1f} MWh")
print(f"  Thermal Storage:         {capacity_thermalstorage:.1f} MWh")


# =============================================================================
# EXTRACT TIMESERIES FROM RESULTS
# =============================================================================

pv_flow        = results[(pv, bus_electricity)]['sequences']['flow']
wind_flow      = results[(wind, bus_electricity)]['sequences']['flow']

el_input       = results[(bus_electricity, electrolyser)]['sequences']['flow']
el_output      = results[(electrolyser, bus_H2)]['sequences']['flow']
el_output_heat = results[(electrolyser, bus_heat)]['sequences']['flow']

fuelcell_input        = results[(bus_H2, fuelcell)]['sequences']['flow']
fuelcell_output_el    = results[(fuelcell, bus_electricity)]['sequences']['flow']
fuelcell_output_heat  = results[(fuelcell, bus_heat)]['sequences']['flow']

heatpump_input  = results[(bus_electricity, heatpump)]['sequences']['flow']
heatpump_output = results[(heatpump, bus_heat)]['sequences']['flow']

el_demand_ts   = results[(bus_electricity, ElectricityDemand)]['sequences']['flow']
h2_demand_ts   = results[(bus_H2, H2Demand)]['sequences']['flow']
heat_demand_ts = results[(bus_heat, HeatDemand)]['sequences']['flow']

curtailment_el   = results[(bus_electricity, electricity_curtailement)]['sequences']['flow']
curtailment_h2   = results[(bus_H2, H2_curtailement)]['sequences']['flow']
curtailment_heat = results[(bus_heat, heat_curtailement)]['sequences']['flow']

battery_soc   = results[(battery, None)]['sequences']['storage_content']
bat_charge    = results[(bus_electricity, battery)]['sequences']['flow']
bat_discharge = results[(battery, bus_electricity)]['sequences']['flow']

saltcavern_soc       = results[(saltcavern, None)]['sequences']['storage_content']
saltcavern_charge    = results[(bus_H2, saltcavern)]['sequences']['flow']
saltcavern_discharge = results[(saltcavern, bus_H2)]['sequences']['flow']

thermalstorage_soc       = results[(thermalstorage, None)]['sequences']['storage_content']
thermalstorage_charge    = results[(bus_heat, thermalstorage)]['sequences']['flow']
thermalstorage_discharge = results[(thermalstorage, bus_heat)]['sequences']['flow']

# Drop the last row (oemof adds an extra timestep due to infer_last_interval)
idx = pv_flow.index[:-1]

pv_flow          = pv_flow.iloc[:-1]
wind_flow        = wind_flow.iloc[:-1]
el_input         = el_input.iloc[:-1]
el_output        = el_output.iloc[:-1]
el_output_heat   = el_output_heat.iloc[:-1]
fuelcell_input        = fuelcell_input.iloc[:-1]
fuelcell_output_el    = fuelcell_output_el.iloc[:-1]
fuelcell_output_heat  = fuelcell_output_heat.iloc[:-1]
heatpump_input   = heatpump_input.iloc[:-1]
heatpump_output  = heatpump_output.iloc[:-1]
el_demand_ts     = el_demand_ts.iloc[:-1]
h2_demand_ts     = h2_demand_ts.iloc[:-1]
heat_demand_ts   = heat_demand_ts.iloc[:-1]
curtailment_el   = curtailment_el.iloc[:-1]
curtailment_h2   = curtailment_h2.iloc[:-1]
curtailment_heat = curtailment_heat.iloc[:-1]
battery_soc          = battery_soc.iloc[:-1]
bat_charge           = bat_charge.iloc[:-1]
bat_discharge        = bat_discharge.iloc[:-1]
saltcavern_soc       = saltcavern_soc.iloc[:-1]
saltcavern_charge    = saltcavern_charge.iloc[:-1]
saltcavern_discharge = saltcavern_discharge.iloc[:-1]
thermalstorage_soc       = thermalstorage_soc.iloc[:-1]
thermalstorage_charge    = thermalstorage_charge.iloc[:-1]
thermalstorage_discharge = thermalstorage_discharge.iloc[:-1]


invest_sizes = {
    "pv":                    results[(pv, bus_electricity)]["scalars"]["invest"],
    "wind":                  results[(wind, bus_electricity)]["scalars"]["invest"],
    "electrolyser":          results[(bus_electricity, electrolyser)]["scalars"]["invest"],
    "heatpump":              results[(bus_electricity, heatpump)]["scalars"]["invest"],
    "fuelcell":              results[(bus_H2, fuelcell)]["scalars"]["invest"],
    "battery_energy":        results[(battery, None)]["scalars"]["invest"],
    "battery_power":         results[(bus_electricity, battery)]["scalars"]["invest"],
    "saltcavern_energy":     results[(saltcavern, None)]["scalars"]["invest"],
    "saltcavern_power":      results[(bus_H2, saltcavern)]["scalars"]["invest"],
    "thermalstorage_energy": results[(thermalstorage, None)]["scalars"]["invest"],
    "thermalstorage_power":  results[(bus_heat, thermalstorage)]["scalars"]["invest"],  
}


# =============================================================================
# SANKEY DIAGRAM — Annual energy flows
# =============================================================================



# ── Node definitions ──────────────────────────────────────────────────────────
# Index:  0=PV  1=Wind  2=Electricity bus  3=Electrolyser  4=H2 bus
#         5=fuelcell  6=Heat bus  7=Heat Pump  8=Battery  9=Salt Cavern
#         10=Thermal Storage  11=El Demand  12=H2 Demand  13=Heat Demand
#         14=El Curtailment  15=H2 Curtailment  16=Heat Curtailment

node_labels = [
    "PV", "Wind", "Electricity bus", "Electrolyser", "H₂ bus",
    "Fuel Cell", "Heat bus", "Heat Pump", "Battery",
    "Salt Cavern", "Thermal Storage",
    "Electricity demand", "H₂ demand", "Heat demand",
    "El curtailment", "H₂ curtailment", "Heat curtailment",
]


node_colors = [
    "#F5C542",  # 0  PV
    "#5AB2FF",  # 1  Wind
    "#378ADD",  # 2  Electricity bus
    "#1D9E75",  # 3  Electrolyser
    "#7F77DD",  # 4  H2 bus
    "#D85A30",  # 5  fuelcell
    "#E2924A",  # 6  Heat bus
    "#EF9F27",  # 7  Heat Pump
    "#9B8AE0",  # 8  Battery
    "#8B6B3D",  # 9  Salt Cavern
    "#E28C3B",  # 10 Thermal Storage
    "#2C7BB6",  # 11 El Demand
    "#5E4FA2",  # 12 H2 Demand
    "#D7191C",  # 13 Heat Demand
    "#BBBBBB",  # 14 El Curtailment
    "#CCCCCC",  # 15 H2 Curtailment
    "#DDDDDD",  # 16 Heat Curtailment
]




flow_specs = [
    # (source, target, series, hover label)
    (0,  2,  pv_flow,              "PV → El bus"),
    (1,  2,  wind_flow,            "Wind → El bus"),

    (2,  3,  el_input,             "El bus → Electrolyser"),
    (2,  7,  heatpump_input,       "El bus → Heat Pump"),
    (2,  8,  bat_charge,           "El bus → Battery (charge)"),
    (2,  11, el_demand_ts,         "El bus → El Demand"),
    (2,  14, curtailment_el,       "El bus → El Curtailment"),

    (5,  2,  fuelcell_output_el,   "fuelcell → El bus"),
    (8,  2,  bat_discharge,        "Battery (discharge) → El bus"),

    (3,  4,  el_output,            "Electrolyser → H₂ bus"),
    (3,  6,  el_output_heat,       "Electrolyser → Heat bus (waste)"),

    (4,  5,  fuelcell_input,            "H₂ bus → fuelcell"),
    (4,  9,  saltcavern_charge,    "H₂ bus → Salt Cavern (charge)"),
    (4,  12, h2_demand_ts,         "H₂ bus → H₂ Demand"),
    (4,  15, curtailment_h2,       "H₂ bus → H₂ Curtailment"),

    (9,  4,  saltcavern_discharge, "Salt Cavern (discharge) → H₂ bus"),

    (7,  6,  heatpump_output,      "Heat Pump → Heat bus"),
    (5,  6,  fuelcell_output_heat,      "fuelcell → Heat bus"),
    (6,  10, thermalstorage_charge,    "Heat bus → Thermal Storage (charge)"),
    (6,  13, heat_demand_ts,       "Heat bus → Heat Demand"),
    (6,  16, curtailment_heat,     "Heat bus → Heat Curtailment"),

    (10, 6,  thermalstorage_discharge, "Thermal Storage (discharge) → Heat bus"),
]

src, tgt, val, lbl = func.build_sankey_flows(flow_specs)

fig_sankey = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=18,
        thickness=22,
        line=dict(color="white", width=0.5),
        label=node_labels,
        color=node_colors,
    ),
    link=dict(
        source=src, target=tgt, value=val,
        label=lbl,
        color=[f"rgba({int(node_colors[s][1:3],16)},{int(node_colors[s][3:5],16)},{int(node_colors[s][5:7],16)},0.33)" for s in src],
    ),
))

fig_sankey.update_layout(
    font=dict(
        family="Arial",
        size=12,
        color="black"
    ),
    height=620,
    margin=dict(l=20, r=20, t=50, b=20),
    paper_bgcolor="white",
    plot_bgcolor="white",
)



# Define paths
base_dir = Path(__file__).resolve().parent
figures_path = base_dir.parent / "[03] Figures"
# Ensure directory exists
figures_path.mkdir(parents=True, exist_ok=True)

# File paths
sankey_html_path = figures_path / "GRAPH_sankey_annual_flows.html"
sankey_pdf_path = figures_path / "GRAPH_sankey_annual_flows.pdf"

# Save Sankey HTML
fig_sankey.write_html(str(sankey_html_path))
print(f"Sankey HTML saved to: {sankey_html_path}")
# Optional: show inline if running interactively
fig_sankey.show()
# Convert HTML to PDF using Playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport={"width": 1400, "height": 700}
    )
    # Open local HTML file
    page.goto(sankey_html_path.as_uri())
    # Allow time for Plotly rendering
    page.wait_for_timeout(2000)
    # Export PDF
    page.pdf(
        path=str(sankey_pdf_path),
        width="1400px",
        height="700px"
    )
    browser.close()
print(f"Sankey PDF saved to: {sankey_pdf_path}")


# =============================================================================
# PLOT 1 — Monthly energy flows
# =============================================================================

monthly = pd.DataFrame({
    "PV":                    pv_flow.values,
    "Wind":                  wind_flow.values,
    "Electrolyser":          el_input.values,
    "Heat Pump":             heatpump_input.values,
    "Fuel Cell":             fuelcell_input.values,
    "Electricity Curtailment": curtailment_el.values,
    "H2 Curtailment":        curtailment_h2.values,
    "Heat Curtailment":      curtailment_heat.values,
    "Electricity Demand":    el_demand_ts.values,
    "H2 Demand":             h2_demand_ts.values,
    "Heat Demand":           heat_demand_ts.values,
}, index=idx).resample("ME").sum()

fig, ax = plt.subplots(figsize=(14, 5))
monthly[["PV", "Wind", "Electrolyser", "Heat Pump", "Fuel Cell",
         "Electricity Curtailment", "H2 Curtailment", "Heat Curtailment"]].plot(
    kind="bar", ax=ax, width=0.8
)
ax.set_ylabel("Energy / [MWh]")
ax.set_xlabel("")
ax.set_xticklabels([d.strftime("%b") for d in monthly.index], rotation=45)
ax.legend(loc="best")
plt.tight_layout()
func.savefigure(fig, "MODEL_monthly_flows")
#plt.show()




# ── Visualization Alternative  ────────────────────────────────────────
monthly_flex_el = pd.DataFrame({
    "PV":               pv_flow.values,  
    "Wind":             wind_flow.values,
    "Fuel Cell (H2→El)":      fuelcell_output_el.values,
    "Battery":          bat_discharge.values,
    "Curtailment":      -curtailment_el.values,
}, index=idx).resample("ME").sum()

monthly_flex_h2 = pd.DataFrame({
    "Electrolyser":     el_output.values,  
    "Salt Cavern":      saltcavern_discharge.values,
    "Curtailment":      -curtailment_h2.values,
}, index=idx).resample("ME").sum()

monthly_flex_heat = pd.DataFrame({
    "Heat Pump":        heatpump_output.values,
    "Fuel Cell heat":         fuelcell_output_heat.values,
    "Electrolyser heat":el_output_heat.values,
    "Thermal Storage":  thermalstorage_discharge.values,
    "Curtailment":      -curtailment_heat.values,
}, index=idx).resample("ME").sum()

month_labels = [d.strftime("%b") for d in monthly_flex_el.index]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for ax, df_monthly, demand_monthly, label in zip(
    axes,
    [monthly_flex_el, monthly_flex_h2, monthly_flex_heat],
    [
        pd.Series(el_demand_ts.values,   index=idx).resample("ME").sum(),
        pd.Series(h2_demand_ts.values,   index=idx).resample("ME").sum(),
        pd.Series(heat_demand_ts.values, index=idx).resample("ME").sum(),
    ],
    ["Electricity", "Hydrogen", "Heat"],
):
    pos_cols = [c for c in df_monthly.columns if c != "Curtailment"]
    df_monthly[pos_cols].plot(kind="bar", ax=ax, stacked=True, width=0.8, alpha=0.85)
    df_monthly[["Curtailment"]].plot(kind="bar", ax=ax, stacked=True, width=0.8,
                                     alpha=0.85, color="firebrick")
    ax.plot(range(len(demand_monthly)), demand_monthly.values,
            "k--o", lw=1.5, ms=4, label="Demand", zorder=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{label} Bus\nMonthly Flexibility Stack")
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel("Energy / [MWh]")
    ax.legend(fontsize=7, loc="upper right")

plt.suptitle("Monthly Flexibility Provision by Source and Bus", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "MODEL_monthly_flows_stacked")
#plt.show()



# =============================================================================
# PLOT 2 — Sample week: hourly dispatch for all 3 buses
# =============================================================================

sl = slice("2023-01-10", "2023-01-17")

# ── Electricity bus ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

windplot = wind_flow.loc[sl]
pvplot   = pv_flow.loc[sl]
fuelcellplot  = fuelcell_output_el.loc[sl]
x        = windplot.index
base     = np.zeros_like(windplot.values)

axes[0].fill_between(x, base, base + windplot.values,                  label="Wind",                  color="tab:blue",   alpha=0.7, edgecolor='none')
base += windplot.values
axes[0].fill_between(x, base, base + pvplot.values,                    label="PV",                    color="tab:orange", alpha=0.7, edgecolor='none')
base += pvplot.values
axes[0].fill_between(x, base, base + fuelcellplot.values,                   label="Fuel Cell electricity output", color="tab:green",  alpha=0.7, edgecolor='none')
axes[0].plot(el_input.loc[sl].index,       el_input.loc[sl].values,       label="Electrolyser input",    color="black",  lw=1.5)
axes[0].plot(heatpump_input.loc[sl].index, heatpump_input.loc[sl].values, label="Heat Pump input",       color="purple", lw=1.5)
axes[0].plot(el_demand_ts.loc[sl].index,   el_demand_ts.loc[sl].values,   label="Electricity demand",    color="orange", lw=1.5, ls="--")
axes[0].plot(curtailment_el.loc[sl].index, curtailment_el.loc[sl].values, label="Curtailment",           color="red",    lw=1,   ls="--")
axes[0].set_ylabel("MWh/h")
axes[0].legend(loc="upper right")

axes[1].plot(wind_flow.loc[sl].index, wind_flow.loc[sl].values + pv_flow.loc[sl].values + fuelcell_output_el.loc[sl].values, label="Electricity produced", color="green",  lw=1.5)
axes[1].plot(el_demand_ts.loc[sl].index, el_demand_ts.loc[sl].values, label="Electricity demand", color="orange", lw=1.5, ls="--")
axes[1].set_ylabel("MWh/h")
axes[1].legend(loc="upper right")

axes[2].fill_between(battery_soc.loc[sl].index, battery_soc.loc[sl].values, alpha=0.5, color="purple", label="Battery SOC")
axes[2].plot(bat_charge.loc[sl].index,    bat_charge.loc[sl].values,    label="Charging",    color="blue", lw=1)
axes[2].plot(bat_discharge.loc[sl].index, bat_discharge.loc[sl].values, label="Discharging", color="red",  lw=1)
axes[2].set_ylabel("MWh")
axes[2].legend(loc="upper right")

plt.tight_layout()
func.savefigure(fig, "MODEL_sample_week_dispatch_ElectricityBus")
#plt.show()

# ── H2 bus ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

el_output_plot = el_output.loc[sl]
x    = el_output_plot.index
base = np.zeros_like(el_output_plot.values)

axes[0].fill_between(x, base, base + el_output_plot.values, label="Electrolyser Output", color="tab:blue", alpha=0.7, edgecolor='none')
axes[0].plot(fuelcell_input.loc[sl].index,      fuelcell_input.loc[sl].values,      label="Fuel Cell input",      color="black",  lw=1.5)
axes[0].plot(h2_demand_ts.loc[sl].index,   h2_demand_ts.loc[sl].values,   label="H2 demand",      color="orange", lw=1.5, ls="--")
axes[0].plot(curtailment_h2.loc[sl].index, curtailment_h2.loc[sl].values, label="H2 Curtailment", color="red",    lw=1,   ls="--")
axes[0].set_ylabel("MWh/h")
axes[0].legend(loc="upper right")

axes[1].plot(el_output_plot.index, el_output_plot.values, label="H2 produced", color="green",  lw=1.5)
axes[1].plot(h2_demand_ts.loc[sl].index, h2_demand_ts.loc[sl].values, label="H2 demand", color="orange", lw=1.5, ls="--")
axes[1].set_ylabel("MWh/h")
axes[1].legend(loc="upper right")

axes[2].fill_between(saltcavern_soc.loc[sl].index, saltcavern_soc.loc[sl].values, alpha=0.5, color="purple", label="Salt Cavern SOC")
axes[2].plot(saltcavern_charge.loc[sl].index,    saltcavern_charge.loc[sl].values,    label="Charging",    color="blue", lw=1)
axes[2].plot(saltcavern_discharge.loc[sl].index, saltcavern_discharge.loc[sl].values, label="Discharging", color="red",  lw=1)
axes[2].set_ylabel("MWh")
axes[2].legend(loc="upper right")

plt.tight_layout()
func.savefigure(fig, "MODEL_sample_week_dispatch_H2Bus")
#plt.show()

# ── Heat bus ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

electrolyser_plot = el_output_heat.loc[sl]
fuelcellplot_heat      = fuelcell_output_heat.loc[sl]
heatpumpplot      = heatpump_output.loc[sl]
x    = electrolyser_plot.index
base = np.zeros_like(electrolyser_plot.values)

axes[0].fill_between(x, base, base + heatpumpplot.values,                 label="Heat Pump Output",          color="tab:blue",   alpha=0.7, edgecolor='none')
base += heatpumpplot.values
axes[0].fill_between(x, base, base + fuelcellplot_heat.values,                 label="Fuel Cell heat output",           color="tab:green",  alpha=0.7, edgecolor='none')
base += fuelcellplot_heat.values
axes[0].fill_between(x, base, base + electrolyser_plot.values,            label="Electrolyser heat output",  color="tab:purple", alpha=0.7, edgecolor='none')
axes[0].plot(heat_demand_ts.loc[sl].index,   heat_demand_ts.loc[sl].values,   label="Heat demand",       color="orange", lw=1.5, ls="--")
axes[0].plot(curtailment_heat.loc[sl].index, curtailment_heat.loc[sl].values, label="Heat Curtailment",  color="red",    lw=1,   ls="--")
axes[0].set_ylabel("MWh/h")
axes[0].legend(loc="upper right")

axes[1].plot(heatpumpplot.index, heatpumpplot.values + fuelcellplot_heat.values + electrolyser_plot.values, label="Heat produced", color="green",  lw=1.5)
axes[1].plot(heat_demand_ts.loc[sl].index, heat_demand_ts.loc[sl].values, label="Heat demand", color="orange", lw=1.5, ls="--")
axes[1].set_ylabel("MWh/h")
axes[1].legend(loc="upper right")

axes[2].fill_between(thermalstorage_soc.loc[sl].index, thermalstorage_soc.loc[sl].values, alpha=0.5, color="purple", label="Thermal Storage SOC")
axes[2].plot(thermalstorage_charge.loc[sl].index,    thermalstorage_charge.loc[sl].values,    label="Charging",    color="blue", lw=1)
axes[2].plot(thermalstorage_discharge.loc[sl].index, thermalstorage_discharge.loc[sl].values, label="Discharging", color="red",  lw=1)
axes[2].set_ylabel("MWh")
axes[2].legend(loc="upper right")

plt.tight_layout()
func.savefigure(fig, "MODEL_sample_week_dispatch_HeatBus")
#plt.show()


# =============================================================================
# PLOT F3 — Sector-coupling dispatch: where does electricity go each hour?
# Purpose : Shows how the electrolyser and heat pump compete for electricity,
#           and how the fuelcell feeds electricity back. The stacked area reveals
#           which sector-coupling pathway dominates at each time of year.
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# ── Top: electricity generation and use ───────────────────────────────────────
gen_total = wind_flow + pv_flow + fuelcell_output_el
base = np.zeros(len(gen_total))

axes[0].fill_between(gen_total.index, base, base + wind_flow.values,
                     label="Wind", color="tab:blue", alpha=0.75)
base += wind_flow.values
axes[0].fill_between(gen_total.index, base, base + pv_flow.values,
                     label="PV", color="tab:orange", alpha=0.75)
base += pv_flow.values
axes[0].fill_between(gen_total.index, base, base + fuelcell_output_el.values,
                     label="Fuel Cell → Electricity", color="tab:green", alpha=0.75)

axes[0].plot(el_demand_ts.index,   el_demand_ts.values,   color="black",  lw=1.2, label="El demand")
axes[0].plot(el_input.index,       el_input.values,       color="purple", lw=1.0, label="Electrolyser")
axes[0].plot(heatpump_input.index, heatpump_input.values, color="red",    lw=1.0, label="Heat Pump")
axes[0].set_ylabel("MW")
axes[0].legend(loc="upper right", fontsize=7, ncol=3)
axes[0].set_title("Electricity Bus: Generation and Sector-Coupling Dispatch")

# ── Middle: H2 bus ────────────────────────────────────────────────────────────
base = np.zeros(len(el_output))
axes[1].fill_between(el_output.index, base, base + el_output.values,
                     label="Electrolyser → H₂", color="tab:blue", alpha=0.75)
axes[1].plot(fuelcell_input.index,    fuelcell_input.values,    color="black",  lw=1.2, label="Fuel Cell demand (H₂)")
axes[1].plot(h2_demand_ts.index, h2_demand_ts.values, color="orange", lw=1.2, label="H₂ end demand")
axes[1].set_ylabel("MW")
axes[1].legend(loc="upper right", fontsize=7, ncol=3)
axes[1].set_title("H₂ Bus: Production and Consumption")

# ── Bottom: heat bus ──────────────────────────────────────────────────────────
base = np.zeros(len(heatpump_output))
axes[2].fill_between(heatpump_output.index, base, base + heatpump_output.values,
                     label="Heat Pump", color="tab:red", alpha=0.75)
base += heatpump_output.values
axes[2].fill_between(el_output_heat.index, base, base + el_output_heat.values,
                     label="Electrolyser waste heat", color="tab:purple", alpha=0.75)
base += el_output_heat.values
axes[2].fill_between(fuelcell_output_heat.index, base, base + fuelcell_output_heat.values,
                     label="Fuel Cell heat", color="tab:green", alpha=0.75)
axes[2].plot(heat_demand_ts.index, heat_demand_ts.values, color="black", lw=1.2, ls="--", label="Heat demand")
axes[2].set_ylabel("MW")
axes[2].legend(loc="upper right", fontsize=7, ncol=2)
axes[2].set_title("Heat Bus: Production and Demand")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[2].set_xlabel("Month")

plt.suptitle("Sector-Coupling Dispatch Across All Three Energy Buses", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F3_sector_coupling_dispatch_full_year")
# plt.show()


# =============================================================================
# PLOT 3 — Twin x-axis + Twin y-axis figures
#   Bottom x: calendar months (time series)
#   Top x:    % of year ranked (duration curve)
#   Left y:   actual values [MW] or [MWh]
#   Right y:  normalised 0–1 (fraction of capacity)
#   Figure 1: Storage SOC | Figure 2: PtX Load
#   Each figure: 3 rows × 1 col, (a)(b)(c) labelled
# =============================================================================
storage_colors = ["purple", "brown", "orange"]

cf_electrolyser = func.capacity_factor(el_input, capacity_electrolyser, dt) if capacity_electrolyser > 0 else 0.0
cf_fuelcell = func.capacity_factor(fuelcell_input, capacity_fuelcell, dt) if capacity_fuelcell > 0 else 0.0
cf_heatpump = func.capacity_factor(heatpump_input, capacity_heatpump, dt) if capacity_heatpump > 0 else 0.0
cf_battery = func.capacity_factor(battery_soc,   capacity_battery, dt) if capacity_battery > 0 else 0.0
cf_saltcavern = func.capacity_factor(saltcavern_soc, capacity_saltcavern, dt) if capacity_saltcavern > 0 else 0.0
cf_thermalstorage = func.capacity_factor(thermalstorage_soc, capacity_thermalstorage, dt) if capacity_thermalstorage > 0 else 0.0

'''for i, (label, flow, capacity, cf, fname) in enumerate([
    ("Electrolyser Load / [MW]", el_input,       capacity_electrolyser, cf_electrolyser, "MODEL_load_duration_curve_electrolyser"),
    ("Fuel Cell Load / [MW]",          fuelcell_input,       capacity_fuelcell,          cf_fuelcell,          "MODEL_load_duration_curve_fuelcell"),
    ("Heat Pump Load / [MW]",    heatpump_input,  capacity_heatpump,     cf_heatpump,     "MODEL_load_duration_curve_heatpump"),
    ("Battery SOC / [MWh]",      battery_soc,     capacity_battery,      cf_battery,      "MODEL_battery_soc_year"),
    ("Salt Cavern SOC / [MWh]",  saltcavern_soc,  capacity_saltcavern,   cf_saltcavern,   "MODEL_saltcavern_soc_year"),
    ("Thermal Storage SOC / [MWh]", thermalstorage_soc, capacity_thermalstorage, cf_thermalstorage, "MODEL_thermal_storage_soc_year"),
]):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # --- Time series plot ---
    if i >= 3:
        color = storage_colors[i - 3]
        axes[0].fill_between(flow.index, flow.values, 0, color=color, alpha=0.4)
    else:
        axes[0].plot(flow.index, flow.values, color="red", lw=0.8)

    axes[0].axhline(capacity, color="black", ls="--", lw=1, label="Installed capacity")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel(label)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].legend()

    # --- Load duration curve ---
    sorted_load = flow.sort_values(ascending=False).values
    axes[1].plot(sorted_load/capacity, color="steelblue", lw=1.5)
    if i >= 3:
        axes[1].axhline(cf, color="black", ls="--", lw=1, label=f"Avg SOC: {cf:.2f}")
    else:
        axes[1].axhline(cf, color="black", ls="--", lw=1, label=f"Capacity Factor: {cf:.2f}")
    axes[1].set_xlabel("Hours per year (ranked)")
    axes[1].set_ylabel(label)
    axes[1].legend()

    plt.tight_layout()
    func.savefigure(fig, fname)
    plt.show()'''

# ── Data ──────────────────────────────────────────────────────────────────────
storage_data = [
    ("Battery SOC / [GWh]",         battery_soc/1e3,        capacity_battery/1e3,        cf_battery,        "purple"),
    ("Salt Cavern SOC / [GWh]",      saltcavern_soc/1e3,     capacity_saltcavern/1e3,     cf_saltcavern,     "brown"),
    #("Thermal Storage SOC / [GWh]",  thermalstorage_soc/1e3, capacity_thermalstorage/1e3, cf_thermalstorage, "y"),
]
ptx_data = [
    ("Electrolyser Load / [MW]", el_input,      capacity_electrolyser, cf_electrolyser, "tab:blue"),
    ("Fuel Cell Load / [MW]",    fuelcell_input, capacity_fuelcell,    cf_fuelcell,     "gold"),
    ("Heat Pump Load / [MW]",    heatpump_input, capacity_heatpump,    cf_heatpump,     "tab:red"),
]

func.plot_combined_twinxy(
    storage_data, is_storage=True,
    fname="MODEL_storage_soc_combined",
)

func.plot_combined_twinxy(
    ptx_data, is_storage=False,
    fname="MODEL_ptx_load_combined",
)



# =============================================================================
# PLOT F6 — Storage cycling: charge vs discharge scatter
# Purpose : Each point is one hour. Clustering near the axes indicates the
#           storage mostly charges OR discharges (not both simultaneously,
#           which would be a model artefact). The spread shows flexibility range.
# =============================================================================

# filter by ENERGY capacity, but plot POWER ratios
storage_specs = [
    (bat_charge,            bat_discharge,            invest_sizes["battery_power"],       invest_sizes["battery_energy"],       "Battery",        "purple"),
    (saltcavern_charge,     saltcavern_discharge,     invest_sizes["saltcavern_power"],     invest_sizes["saltcavern_energy"],     "Salt Cavern",    "brown"),
    (thermalstorage_charge, thermalstorage_discharge, invest_sizes["thermalstorage_power"], invest_sizes["thermalstorage_energy"], "Thermal Storage","orange"),
]
active = [
    (ch, dis, p_cap, lbl, col)
    for ch, dis, p_cap, e_cap, lbl, col in storage_specs
    if e_cap > 0
]

n = len(active)

with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 20,
    "axes.titlesize": 16,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": True,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, axes = plt.subplots(1, n, figsize=(15.5, 5), sharey=True)
    if n == 1:
        axes = [axes]

    bins = np.linspace(-1, 1, 101)

    for col, (charge, discharge, p_cap, label, color) in enumerate(active):
        ax = axes[col]

        charge_ratio    =  charge.values / p_cap
        discharge_ratio = -np.abs(discharge.values) / p_cap

        total_hours     = len(charge_ratio)
        pct_charging    = (charge_ratio    != 0).sum() / total_hours * 100
        pct_discharging = (discharge_ratio != 0).sum() / total_hours * 100

        ax.hist(charge_ratio[charge_ratio != 0],
                bins=bins, color=color, alpha=0.80, edgecolor="none",
                label=f"Charging ({pct_charging:.1f}% of hours, max = {charge_ratio.max():.2f})")
        ax.hist(discharge_ratio[discharge_ratio != 0],
                bins=bins, color="steelblue", alpha=0.80, edgecolor="none",
                label=f"Discharging ({pct_discharging:.1f}% of hours, max = {discharge_ratio.min():.2f})")

        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"{label} power ratio / [-]")   # name in x-axis, no title

        # ── (a)(b)(c) label — top left ────────────────────────────────────────
        ax.text(0.02, 0.98, f"({chr(97 + col)})",
                transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top", ha="left", zorder=10)

        if col == 0:
            ax.set_ylabel("Hours")

        if col == 0:
            ax.set_ylabel("Hours")

        # legend below the subplot, outside the plot area
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),   # below x-axis label
            ncol=1,
            borderaxespad=0,
            frameon=True,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)       # make room for the external legends
    func.savefigure(fig, "FLEX_F6_storage_charge_discharge_histogram")
    #plt.show()



# =============================================================================
# PLOT 7 — levelized Cost calculations and cost attribution
# =============================================================================

# ── 1. Annual energy per carrier ──────────────────────────────────────────────
total_h2_mwh    = h2_demand_ts.sum()                                   # MWh_H2/yr
total_h2_kg     = total_h2_mwh / par.H2_CALORIFIC_VALUE_LHV        # kg_H2/yr
total_el_mwh    = el_demand_ts.sum() #+ curtailment_el.sum()          MWh_el/yr (served + curtailed)
total_heat_mwh  = heat_demand_ts.sum()  # MWh_th/yr
total_energy_mwh = total_h2_mwh + total_el_mwh + total_heat_mwh    # MWh_total/yr

# ── 2. Total system cost ──────────────────────────────────────────────────────
total_cost = model.objective()  # €/yr

# ── 3. ep_costs map (annualised CAPEX + fixed OPEX per unit capacity) ─────────
ep_costs_map = {
    "pv": (
        economics.annuity(capex=par.PV_CAPEX_PANELS,   n=par.PV_LIFETIME_SYSTEM, wacc=par.WACC)
        + economics.annuity(capex=par.PV_CAPEX_INVERTER, n=par.PV_LIFETIME_SYSTEM, wacc=par.WACC, u=par.PV_LIFETIME_INVERTER)
        + par.PV_OPEX
    ),
    "wind": (
        economics.annuity(capex=par.WIND_CAPEX, n=par.WIND_LIFETIME_SYSTEM, wacc=par.WACC)
        + par.WIND_OPEX
    ),
    "electrolyser": (
        economics.annuity(capex=par.ELECTROLYSER_CAPEX_SYSTEM, n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC)
        + economics.annuity(capex=par.ELECTROLYSER_CAPEX_STACK,  n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC, u=par.ELECTROLYSER_LIFETIME_STACK)
        + par.ELECTROLYSER_OPEX
    ),
    "heatpump": (
        economics.annuity(capex=par.HEATPUMP_CAPEX, n=par.HEATPUMP_LIFETIME, wacc=par.WACC)
        + par.HEATPUMP_OPEX_FIX
    ),
    "fuelcell": (
        economics.annuity(capex=par.FUELCELL_CAPEX_SYSTEM, n=par.FUELCELL_LIFETIME_SYSTEM, wacc=par.WACC)
        + economics.annuity(capex=par.FUELCELL_CAPEX_STACK,  n=par.FUELCELL_LIFETIME_SYSTEM, wacc=par.WACC, u=par.FUELCELL_LIFETIME_STACK)
        + par.FUELCELL_OPEX_FIX
    ),
    "battery_energy": (
        economics.annuity(capex=par.BATTERY_CAPEX_SYSTEM,       n=par.LIFETIME, wacc=par.WACC, u=par.BATTERY_LIFETIME_SYSTEM)
        + economics.annuity(capex=par.BATTERY_CAPEX_BATTERY_PACK, n=par.LIFETIME, wacc=par.WACC, u=par.BATTERY_LIFETIME_BATTERY_PACK)
    ),
    "battery_power": (
        economics.annuity(capex=par.BATTERY_CAPEX_POWER, n=par.LIFETIME, wacc=par.WACC, u=par.BATTERY_LIFETIME_SYSTEM)
        + par.BATTERY_OPEX_FIX
    ),
    "saltcavern_energy": (
        economics.annuity(capex=par.SALTCAVERN_CAPEX,             n=par.SALTCAVERN_LIFETIME, wacc=par.WACC)
        + economics.annuity(capex=par.SALTCAVERN_CAPEX_CUSHION_GAS, n=par.SALTCAVERN_LIFETIME, wacc=par.WACC)
        + par.SALTCAVERN_OPEX
    ),
    "saltcavern_power": (
        economics.annuity(capex=par.SALTCAVERN_CAPEX_COMPRESSOR, n=par.SALTCAVERN_LIFETIME, wacc=par.WACC, u=par.SALTCAVERN_LIFETIME_COMPRESSOR)
    ),
    "thermalstorage_energy": (
        economics.annuity(capex=par.THERMALSTORAGE_CAPEX, n=par.THERMALSTORAGE_LIFETIME, wacc=par.WACC)
        + par.THERMALSTORAGE_OPEX
    ),
    "thermalstorage_power": 0.0,  # No separate power cost for thermal storage
}

variable_costs_total = {
    "heatpump_variable":   (heatpump_input   * dt * par.HEATPUMP_OPEX_VAR).sum(),
    "battery_charge":      (bat_charge       * dt * par.BATTERY_OPEX_VAR / 2).sum(),
    "battery_discharge":   (bat_discharge    * dt * par.BATTERY_OPEX_VAR / 2).sum(),
    "saltcavern_compress": (saltcavern_charge * dt * par.SALTCAVERN_OPEX_COMPRESSOR).sum(),
}

# ── 4. Fixed cost per technology ──────────────────────────────────────────────
cost_pv             = ep_costs_map["pv"]              * invest_sizes["pv"]
cost_wind           = ep_costs_map["wind"]            * invest_sizes["wind"]
cost_electrolyser   = ep_costs_map["electrolyser"]    * invest_sizes["electrolyser"]
cost_heatpump       = ep_costs_map["heatpump"]        * invest_sizes["heatpump"]
cost_fuelcell            = ep_costs_map["fuelcell"]             * invest_sizes["fuelcell"]
cost_battery        = (ep_costs_map["battery_energy"] * invest_sizes["battery_energy"]
                     + ep_costs_map["battery_power"]  * invest_sizes["battery_power"])
cost_saltcavern     = (ep_costs_map["saltcavern_energy"] * invest_sizes["saltcavern_energy"]
                     + ep_costs_map["saltcavern_power"]  * invest_sizes["saltcavern_power"])
cost_thermalstorage = (ep_costs_map["thermalstorage_energy"] * invest_sizes["thermalstorage_energy"] 
                     + ep_costs_map["thermalstorage_power"]  * invest_sizes["thermalstorage_power"])

# ── 5. Variable cost per technology ───────────────────────────────────────────
cost_var_heatpump   = variable_costs_total["heatpump_variable"]
cost_var_battery    = variable_costs_total["battery_charge"] + variable_costs_total["battery_discharge"]
cost_var_saltcavern = variable_costs_total["saltcavern_compress"]

# ── 6. Electrolyser cost split: H2 output vs heat output ─────────────────────
# The electrolyser produces both H2 and recoverable heat.
# Its cost is split proportionally to each output's energy contribution.
electrolyser_total_output = el_output.sum() + el_output_heat.sum()
if electrolyser_total_output > 0:
    electro_frac_h2   = el_output.sum()      / electrolyser_total_output
    electro_frac_heat = el_output_heat.sum() / electrolyser_total_output
else:
    electro_frac_h2   = 1.0
    electro_frac_heat = 0.0

cost_electrolyser_h2   = cost_electrolyser * electro_frac_h2
cost_electrolyser_heat = cost_electrolyser * electro_frac_heat

# ── 7. fuelcell cost split: electricity output vs heat output ─────────────────────
fuelcell_total_output = fuelcell_output_el.sum() + fuelcell_output_heat.sum()
if fuelcell_total_output > 0:
    fuelcell_frac_el   = fuelcell_output_el.sum()   / fuelcell_total_output
    fuelcell_frac_heat = fuelcell_output_heat.sum() / fuelcell_total_output
else:
    fuelcell_frac_el   = 0.5
    fuelcell_frac_heat = 0.5

cost_fuelcell_el   = cost_fuelcell  * fuelcell_frac_el
cost_fuelcell_heat = cost_fuelcell  * fuelcell_frac_heat

# ── 8. Shared supply cost (PV + Wind) allocated by energy fraction ────────────
# Method: each carrier's share of total useful energy delivered.
# Limitation: implicitly assumes 1 MWh is equally valuable across carriers.
# Alternative (input-based) is not used (for example, because fuelcell produces heat from H2,)
# not electricity, making input tracing produce severely distorted LCOH.
cost_supply = cost_pv + cost_wind

frac_h2   = total_h2_mwh   / total_energy_mwh
frac_el   = total_el_mwh   / total_energy_mwh
frac_heat = total_heat_mwh / total_energy_mwh

cost_supply_h2   = cost_supply * frac_h2
cost_supply_el   = cost_supply * frac_el
cost_supply_heat = cost_supply * frac_heat

# ── 9. Total attributed cost per carrier ──────────────────────────────────────
#
# H2   : supply share + electrolyser (H2 fraction) + salt cavern (dedicated to H2)
# Elec : supply share + battery (dedicated to electricity) + fuelcell electricity share
# Heat : supply share + heat pump (dedicated to heat) + thermal storage
#        + electrolyser heat fraction + fuelcell heat share

cost_h2_total   = (cost_supply_h2
                 + cost_electrolyser_h2
                 + cost_saltcavern
                 + cost_var_saltcavern)

cost_el_total   = (cost_supply_el
                 + cost_battery
                 + cost_var_battery
                 + cost_fuelcell_el)

cost_heat_total = (cost_supply_heat
                 + cost_heatpump + cost_var_heatpump
                 + cost_thermalstorage
                 + cost_electrolyser_heat
                 + cost_fuelcell_heat)

cost_system_total = cost_h2_total + cost_el_total + cost_heat_total

# ── Sanity check ──────────────────────────────────────────────────────────────
cost_manual_total = (cost_pv + cost_wind
                   + cost_electrolyser
                   + cost_heatpump + cost_var_heatpump
                   + cost_fuelcell 
                   + cost_battery + cost_var_battery
                   + cost_saltcavern + cost_var_saltcavern
                   + cost_thermalstorage)

print("\n" + "=" * 65)
print("  COST ATTRIBUTION SANITY CHECK")
print("=" * 65)
print(f"  Sum of attributed costs   : {cost_system_total:>14,.0f} €/yr")
print(f"  Manual cost total         : {cost_manual_total:>14,.0f} €/yr")
print(f"  model.objective()         : {total_cost:>14,.0f} €/yr")
print(f"  Diff (attributed vs obj)  : {abs(cost_system_total - total_cost):>14,.0f} €/yr")
print(f"  Diff (manual vs obj)      : {abs(cost_manual_total  - total_cost):>14,.0f} €/yr")
print("=" * 65)

# ── 10. levelized costs ───────────────────────────────────────────────────────
lcoh2_mwh = cost_h2_total   / total_h2_mwh    if total_h2_mwh   > 0 else np.nan
lcoh2_kg  = lcoh2_mwh * par.H2_CALORIFIC_VALUE_LHV
lcoel_mwh  = cost_el_total   / total_el_mwh    if total_el_mwh   > 0 else np.nan
lcoh_mwh  = cost_heat_total / total_heat_mwh  if total_heat_mwh > 0 else np.nan
lcoe_mwh  = total_cost      / total_energy_mwh if total_energy_mwh > 0 else np.nan
'''NOTE: lcoe_mwh could be calculated as follows. It would yield the same result:
lcoe_mwh = ((lcoh2_mwh * total_h2_mwh
            + lcoel_mwh * total_el_mwh 
            + lcoh_mwh * total_heat_mwh)
            / total_energy_mwh
            ) if total_energy_mwh > 0 else np.nan'''

# Replace the existing lcos scalar lines with:
lcos_el_mwh   = (cost_battery + cost_var_battery) / bat_discharge.sum() if bat_discharge.sum() > 0 else np.nan
lcos_h2_mwh   = (cost_saltcavern + cost_var_saltcavern) / saltcavern_discharge.sum() if saltcavern_discharge.sum() > 0 else np.nan
lcos_h2_kg    = lcos_h2_mwh * par.H2_CALORIFIC_VALUE_LHV
lcos_heat_mwh = cost_thermalstorage / thermalstorage_discharge.sum() if thermalstorage_discharge.sum() > 0 else np.nan
lcos_mwh      = (cost_battery + cost_var_battery + cost_saltcavern + cost_var_saltcavern + cost_thermalstorage) / (bat_discharge.sum() + saltcavern_discharge.sum() + thermalstorage_discharge.sum()) if (bat_discharge + saltcavern_discharge + thermalstorage_discharge).sum() > 0 else np.nan


print("\n" + "=" * 65)
print("  Levelized COST RESULTS")
print("=" * 65)
print(f"  Annual H2 produced        : {total_h2_mwh:>12,.1f} MWh_H2/yr")
print(f"  Annual electricity served : {total_el_mwh:>12,.1f} MWh_el/yr")
print(f"  Annual heat produced      : {total_heat_mwh:>12,.1f} MWh_th/yr")
print(f"  Total energy              : {total_energy_mwh:>12,.1f} MWh/yr")
print(f"  Total system cost         : {total_cost:>12,.0f} €/yr")
print(f"  ---")
print(f"  LCOH2                     : {lcoh2_mwh:>12.2f} €/MWh_H2  ({lcoh2_kg:.2f} €/kg)")
print(f"  LCOEl                      : {lcoel_mwh:>12.2f} €/MWh_el")
print(f"  LCOH                      : {lcoh_mwh:>12.2f} €/MWh_th")
print(f"  LCOE (system)             : {lcoe_mwh:>12.2f} €/MWh  (all carriers)")
print(f"  ---")
print(f"  LCOS Battery              : {lcos_el_mwh:>12.2f} €/MWh_el discharged")
print(f"  LCOS Salt Cavern          : {lcos_h2_mwh:>12.2f} €/MWh_H2 discharged  ({lcos_h2_kg:.2f} €/kg_H2 discharged)")
print(f"  LCOS Thermal Storage      : {lcos_heat_mwh:>12.2f} €/MWh_th discharged")
print(f"  LCOS (overall)            : {lcos_mwh:>12.2f} €/MWh discharged (all storage)")

print("=" * 65)

# ── 11. Grouped cost table (fixed vs variable) ────────────────────────────────
group_map = {
    "pv":                    ("PV",              "fixed"),
    "wind":                  ("Wind",            "fixed"),
    "electrolyser":          ("Electrolyser",    "fixed"),
    "heatpump":              ("Heat Pump",       "fixed"),
    "fuelcell":              ("Fuel Cell",             "fixed"),
    "battery_energy":        ("Battery",         "fixed"),
    "battery_power":         ("Battery",         "fixed"),
    "saltcavern_energy":     ("Salt Cavern",     "fixed"),
    "saltcavern_power":      ("Salt Cavern",     "fixed"),
    "thermalstorage_energy": ("Thermal Storage", "fixed"),
    "thermalstorage_power":  ("Thermal Storage", "fixed"),
    "heatpump_variable":     ("Heat Pump",       "variable"),
    "battery_charge":        ("Battery",         "variable"),
    "battery_discharge":     ("Battery",         "variable"),
    "saltcavern_compress":   ("Salt Cavern",     "variable"),
}


print("\n" + "=" * 55)
print("  COST BREAKDOWN")
print("=" * 55)
print(f"\n  {'Component':<35} {'Size':>10}  {'Cost (€/yr)':>12}")
print(f"  {'-'*57}")

total_manual = 0
grouped = defaultdict(lambda: {"fixed": 0.0, "variable": 0.0})

# Fixed costs
for key in ep_costs_map:

    size = invest_sizes[key]
    cost = ep_costs_map[key] * size

    tech, ctype = group_map[key]
    grouped[tech][ctype] += cost

    total_manual += cost
    unit = "MWh" if "energy" in key else "MW"
    print(f"  {key:<35} {size:>8.1f} {unit}  {cost:>12.0f}")

# Variable costs
print(f"\n  Variable costs:")
for key, cost in variable_costs_total.items():
    tech, ctype = group_map[key]
    grouped[tech][ctype] += cost

    total_manual += cost
    print(f"  {key:<35}            {cost:>12.0f}")

print(f"\n  {'Manual total':<35}            {total_manual:>12.0f}")
print(f"  {'model.objective()':<35}            {total_cost:>12.0f}")
print(f"  {'Difference':<35}            {abs(total_manual - total_cost):>12.0f}")
print("=" * 55)

print("\n" + "=" * 80)
print("  FIXED vs VARIABLE COST SPLIT")
print("=" * 80)
for tech, costs in grouped.items():
    fixed    = costs["fixed"]
    variable = costs["variable"]
    total_t  = fixed + variable
    if total_t == 0:
        continue
    print(
        f"{tech:<15} | "
        f"fixed: {fixed:>12,.0f} € ({100*fixed/total_t:>5.1f}%) ; "
        f"variable: {variable:>12,.0f} € ({100*variable/total_t:>5.1f}%)"
    )
print("=" * 80)

# ── 12. Pie charts ────────────────────────────────────────────────────────────
fixed_labels, fixed_values = func.sorted_pie(
    [(tech, costs["fixed"]) for tech, costs in grouped.items()]
)
total_labels, total_values = func.sorted_pie(
    [(tech, costs["fixed"] + costs["variable"]) for tech, costs in grouped.items()]
)
'''
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].pie(fixed_values, labels=fixed_labels, autopct='%1.1f%%', startangle=90)
axes[0].set_title("Fixed Costs Breakdown (Annualised CAPEX + Fixed OPEX)", fontsize=11)
axes[1].pie(total_values, labels=total_labels, autopct='%1.1f%%', startangle=90)
axes[1].set_title("Total Costs Breakdown (Fixed + Variable OPEX)", fontsize=11)
for ax in axes:
    ax.axis('equal')
#plt.suptitle("System Cost Breakdown by Technology", fontsize=13, y=1.01)
plt.tight_layout()
func.savefigure(fig, "MODEL_cost_breakdown_pies")
plt.show()
'''


# Hardcoded order and colors
ordered_labels = ["Wind", "PV", "Electrolyser", "Fuel Cell", "Heat Pump", "Battery", "Salt Cavern"]
ordered_colors = ["skyblue", "peru", "tab:blue", "gold", "tab:red", "purple", "brown"]

label_to_value = dict(zip(total_labels, total_values))
ordered_values = [label_to_value.get(l, 0) for l in ordered_labels]

filtered_pairs = [
    (l, v, c)
    for l, v, c in zip(ordered_labels, ordered_values, ordered_colors)
    if v > 0
]

if filtered_pairs:
    ordered_labels, ordered_values, ordered_colors = map(list, zip(*filtered_pairs))
else:
    ordered_labels, ordered_values, ordered_colors = [], [], []

total = sum(ordered_values)
threshold = 5.0


with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 40,
    "axes.titlesize": 16,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "legend.fontsize": 40,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": True,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, ax = plt.subplots(figsize=(8, 7))

    wedges, texts, autotexts = ax.pie(
        ordered_values,
        labels=None,
        colors=ordered_colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= threshold else "",
        startangle=90,
        pctdistance=0.82,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )

    for autotext in autotexts:
        autotext.set_fontsize(20)
        autotext.set_fontfamily("serif")

    # Collect small wedges and sort by value descending (smallest gets highest k = longest line)
    small_wedges = sorted(
        [(wedge, value) for wedge, value in zip(wedges, ordered_values)
         if value / total * 100 < threshold],
        key=lambda x: x[1], reverse=True
    )

    for k, (wedge, value) in enumerate(small_wedges):
        pct = value / total * 100
        mid = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)

        # Larger k = longer line = smaller slice
        r_elbow = 1.15 + k * 0.12
        r_text  = 1.25 + k * 0.12

        xA, yA = 0.95 * np.cos(mid), 0.95 * np.sin(mid)
        xB, yB = r_elbow * np.cos(mid), r_elbow * np.sin(mid)
        xT, yT = r_text  * np.cos(mid), r_text  * np.sin(mid)

        ax.annotate(
            "",
            xy=(xA, yA), xycoords="data",
            xytext=(xB, yB), textcoords="data",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.8),
        )
        ax.text(
            xT, yT,
            f"{pct:.1f}%",
            ha="left" if xT >= 0 else "right",
            va="center",
            fontsize=20, fontfamily="serif",
        )

    ax.legend(
        wedges,
        ordered_labels,
        #title="Technology",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=20,
        title_fontsize=20,
        frameon=True,
        labelspacing=0.8,
    )

    ax.axis('equal')
    plt.tight_layout()
    func.savefigure(fig, "MODEL_cost_breakdown_pie_total")
    #plt.show()




# ── 13. Carrier cost attribution + levelized cost bar charts ──────────────────
#  Single source of truth for colors 
COMP_COLORS = {
    "Electrolyser":    "tab:blue",
    "Fuel Cell":       "gold",
    "Heat Pump":       "tab:red",
    "Battery":         "purple",
    "Salt Cavern":     "brown",
    "Thermal Storage": "tab:orange",
    "Supply":          "tab:green",
    "Wind":            "skyblue",
    "PV":              "peru",
}

#  Variants that share a color with their base component 
COMP_ALIASES = {
    "Electrolyser (heat)":   "Electrolyser",
    "Fuel Cell (el share)":  "Fuel Cell",
    "Fuel Cell (heat share)":"Fuel Cell",
}



#  carrier_components (unchanged except now colors will resolve cleanly) 
carrier_components = {
    "H2": {
        "Supply": cost_supply_h2,
        "Electrolyser":    cost_electrolyser_h2,
        "Salt Cavern":     cost_saltcavern + cost_var_saltcavern,
    },
    "Electricity": {
        "Supply":      cost_supply_el,
        "Battery":              cost_battery + cost_var_battery,
        "Fuel Cell (el share)": cost_fuelcell_el,
    },
    "Heat": {
        "Supply":        cost_supply_heat,
        "Heat Pump":              cost_heatpump + cost_var_heatpump,
        "Thermal Storage":        cost_thermalstorage,
        "Fuel Cell (heat share)": cost_fuelcell_heat,
        "Electrolyser (heat)":    cost_electrolyser_heat,
    },
}

all_components = sorted({comp for d in carrier_components.values() for comp in d})



# ── Plot 
with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": True,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax_lc = ax.twinx()

    categories = list(carrier_components.keys())
    x       = np.arange(len(categories))
    bottoms = np.zeros(len(categories))

    legend_seen = set()

    for comp in all_components:
        vals = np.array([carrier_components[carrier].get(comp, 0.0) / 1e6
                         for carrier in categories])
        base  = COMP_ALIASES.get(comp, comp)
        color = func.get_color(COMP_ALIASES, COMP_COLORS, comp)

        if base not in legend_seen:
            ax.bar(x, vals, bottom=bottoms, label=base, color=color)
            legend_seen.add(base)
        else:
            ax.bar(x, vals, bottom=bottoms, color=color)

        bottoms += vals

    lc_values = [lcoh2_mwh, lcoel_mwh, lcoh_mwh]
    ax_lc.plot(x, lc_values,
               color="black", lw=1.5, ls="--", marker="o", markersize=5,
               zorder=5, label="Levelized cost")

    ax_lc.set_ylabel("Levelized Cost / [€/MWh]")
    ax_lc.set_ylim(0, max(lc_values) * 1.25)

    ax.set_xticks(x)
    ax.set_xticklabels([r"H$_2$", "Electricity", "Heat"])
    ax.set_ylabel("Annualised Cost / [M€/year]")


    handles_bar, labels_bar = ax.get_legend_handles_labels()
    handles_lc,  labels_lc  = ax_lc.get_legend_handles_labels()
    ax.legend(handles_bar + handles_lc, labels_bar + labels_lc,
              loc="upper left",
              bbox_to_anchor=(0.75, 0.98), borderaxespad=0)

    plt.tight_layout()
    func.savefigure(fig, "MODEL_cost_attribution_waterfall")
    #plt.show()



















# =============================================================================
# PLOT 8 — Instantaneous levelized Costs (cost / demand at each timestep)
# =============================================================================
#
# METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
# Instantaneous LC[t] = hourly_attributed_cost[t] / hourly_demand[t]  [€/MWh]
#
# Fixed costs are spread uniformly (capacity charges cannot be assigned to
# specific dispatch hours). Variable costs are reconstructed hour-by-hour.
#
# Sanity check: cost_ts.sum() / demand_ts.sum() must equal the scalar LC.
# NOTE: np.mean(lc_instantaneous) ≠ scalar LC unless demand is perfectly flat,
#       because hours with near-zero demand produce huge instantaneous values
#       that dominate a simple arithmetic mean. Always use the weighted check.



# ── H2: fixed cost spread uniformly + salt cavern variable cost ───────────────
# Fixed costs attributed to H2: shared supply share + electrolyser H2 frac + salt cavern fixed
cost_h2_fixed_ph  = (cost_supply_h2 + cost_electrolyser_h2 + cost_saltcavern) / len(h2_demand_ts)
saltcavern_var_ts = saltcavern_charge * dt * par.SALTCAVERN_OPEX_COMPRESSOR     # €/h
cost_h2_ts        = pd.Series(cost_h2_fixed_ph, index=h2_demand_ts.index) + saltcavern_var_ts

# ── Electricity: fixed cost spread uniformly + battery + fuelcell-el variable ──────
cost_el_fixed_ph  = (cost_supply_el + cost_battery + cost_fuelcell_el) / len(el_demand_ts)
cost_el_var_ts    = (bat_charge + bat_discharge) * dt * par.BATTERY_OPEX_VAR / 2  # €/h
cost_el_ts        = pd.Series(cost_el_fixed_ph, index=el_demand_ts.index) + cost_el_var_ts

# ── Heat: fixed cost spread uniformly + heat pump + fuelcell-heat variable ─────────
cost_heat_fixed_ph = (cost_supply_heat + cost_heatpump
                    + cost_thermalstorage + cost_electrolyser_heat
                    + cost_fuelcell_heat) / len(heat_demand_ts)


cost_heat_var_ts   = heatpump_input * dt * par.HEATPUMP_OPEX_VAR                # €/h
cost_heat_ts       = pd.Series(cost_heat_fixed_ph, index=heat_demand_ts.index) + cost_heat_var_ts

# ── System LCOE: total hourly cost / total hourly demand ──────────────────────
total_demand_ts    = h2_demand_ts + el_demand_ts + heat_demand_ts               # MWh/h
cost_system_ts     = cost_h2_ts + cost_el_ts + cost_heat_ts                     # €/h
total_demandsafeval  = total_demand_ts.replace(0, np.nan)

# ── Instantaneous LC timeseries ───────────────────────────────────────────────
h2_demandsafeval      = h2_demand_ts.replace(0, np.nan)
el_demandsafeval      = el_demand_ts.replace(0, np.nan)
heat_demandsafeval    = heat_demand_ts.replace(0, np.nan)

lcoh2_instantaneous = cost_h2_ts   / h2_demandsafeval                            # €/MWh_H2
lcoel_instantaneous  = cost_el_ts   / el_demandsafeval                            # €/MWh_el
lcoh_instantaneous  = cost_heat_ts / heat_demandsafeval                          # €/MWh_th
lcoe_instantaneous  = cost_system_ts / total_demandsafeval                       # €/MWh

# ── Battery: fixed cost spread uniformly per hour + hourly variable cost ──────
lcos_el_fixed_ph   = (cost_battery) / len(bat_discharge)
lcos_el_var_ts     = (bat_charge + bat_discharge) * dt * par.BATTERY_OPEX_VAR / 2
lcos_el_cost_ts    = pd.Series(lcos_el_fixed_ph, index=bat_discharge.index) + lcos_el_var_ts


# ── Salt cavern: fixed cost spread uniformly + compressor variable cost ────────
lcos_h2_fixed_ph   = cost_saltcavern / len(saltcavern_discharge)
lcos_h2_var_ts     = saltcavern_charge * dt * par.SALTCAVERN_OPEX_COMPRESSOR
lcos_h2_cost_ts    = pd.Series(lcos_h2_fixed_ph, index=saltcavern_discharge.index) + lcos_h2_var_ts

# ── Thermal storage: fixed cost only, no variable cost ────────────────────────
lcos_heat_fixed_ph = cost_thermalstorage / len(thermalstorage_discharge)
lcos_heat_cost_ts  = pd.Series(lcos_heat_fixed_ph, index=thermalstorage_discharge.index) + 0.0



# ── Sanity checks: cost_ts.sum() / demand_ts.sum() must equal scalar LC ───────
# Do NOT use np.mean() — that is a simple average and will NOT match the scalar
# LC unless demand happens to be perfectly flat.
lcoh2_check = cost_h2_ts.sum()     / h2_demand_ts.sum()
lcoel_check  = cost_el_ts.sum()     / el_demand_ts.sum()
lcoh_check  = cost_heat_ts.sum()   / heat_demand_ts.sum()
lcoe_check  = cost_system_ts.sum() / total_demand_ts.sum()
lcos_el_check   = lcos_el_cost_ts.sum()   / bat_discharge.sum()
lcos_h2_check   = lcos_h2_cost_ts.sum()   / saltcavern_discharge.sum()
lcos_heat_check = lcos_heat_cost_ts.sum() / thermalstorage_discharge.sum()

print("\n" + "=" * 68)
print("  INSTANTANEOUS LC — SANITY CHECKS (cost_ts.sum()/demand_ts.sum())")
print("  NOTE: np.mean(lc_instantaneous) ≠ scalar LC — use weighted sum.")
print("=" * 68)
print(f"  LCOH2  scalar: {lcoh2_mwh:>8.3f}  reconstructed: {lcoh2_check:>8.3f}  Δ={abs(lcoh2_mwh-lcoh2_check):.5f}")
print(f"  LCOEl   scalar: {lcoel_mwh:>8.3f}  reconstructed: {lcoel_check:>8.3f}  Δ={abs(lcoel_mwh-lcoel_check):.5f}")
print(f"  LCOH   scalar: {lcoh_mwh:>8.3f}  reconstructed: {lcoh_check:>8.3f}  Δ={abs(lcoh_mwh-lcoh_check):.5f}")
print(f"  LCOE   scalar: {lcoe_mwh:>8.3f}  reconstructed: {lcoe_check:>8.3f}  Δ={abs(lcoe_mwh-lcoe_check):.5f}")
print(f"  LCOS_el  scalar: {lcos_el_mwh:>8.3f}  reconstructed: {lcos_el_check:>8.3f}  Δ={abs(lcos_el_mwh - lcos_el_check):.5f}")
print(f"  LCOS_h2  scalar: {lcos_h2_mwh:>8.3f}  reconstructed: {lcos_h2_check:>8.3f}  Δ={abs(lcos_h2_mwh - lcos_h2_check):.5f}")
print(f"  LCOS_heat scalar:{lcos_heat_mwh:>8.3f}  reconstructed: {lcos_heat_check:>8.3f}  Δ={abs(lcos_heat_mwh - lcos_heat_check):.5f}")
print("=" * 68)

'''print("Avg LCOH2 (from instantaneous): ", np.mean(lcoh2_instantaneous), "€/MWh")
print("Avg LCOEl (from instantaneous): ", np.mean(lcoel_instantaneous), "€/MWh")
print("Avg LCOH (from instantaneous): ", np.mean(lcoh_instantaneous), "€/MWh")
print("Avg LCOE (from instantaneous): ", np.mean(lcoe_instantaneous), "€/MWh")'''
''' #SANITY CHECK: these sums must match the model objective (total cost)
print("Total cost from timeseries: ", cost_system_ts.sum(), "€")
print("Model objective: ", model.objective(), "€")
RESULT:  THEY DO ✅'''


# ── Plot 8a — Instantaneous LCOH2 ────────────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcoh2_instantaneous,
    demand_ts    = h2_demand_ts,
    annual_lc    = lcoh2_mwh,
    lc_label     = r"LCOH$_2$",
    demand_label = r"H$_2$ Demand",
    lc_color     = "tab:green",
    demand_color = "tab:green",
    fig_name     = "MODEL_lcoh2_instantaneous",
)

# ── Plot 8b — Instantaneous LCOEl ─────────────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcoel_instantaneous,
    demand_ts    = el_demand_ts,
    annual_lc    = lcoel_mwh,
    lc_label     = r"LCOE$_{\mathrm{l}}$",
    demand_label = r"Electricity Demand",
    lc_color     = "tab:olive",
    demand_color = "tab:olive",
    fig_name     = "MODEL_lcoel_instantaneous",
)

# ── Plot 8c — Instantaneous LCOH (heat) ──────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcoh_instantaneous,
    demand_ts    = heat_demand_ts,
    annual_lc    = lcoh_mwh,
    lc_label     = r"LCOH",
    demand_label = r"Heat Demand",
    lc_color     = "tab:orange",
    demand_color = "tab:orange",
    fig_name     = "MODEL_lcoh_instantaneous",
)

# ── Plot 8d — Instantaneous LCOE (system) ────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcoe_instantaneous,
    demand_ts    = total_demand_ts,
    annual_lc    = lcoe_mwh,
    lc_label     = r"LCOE",
    demand_label = r"Total Demand",
    lc_color     = "slateblue",
    demand_color = "slateblue",
    fig_name     = "MODEL_lcoe_instantaneous",
)


# ── Plot 8e — LC duration curves (all 4 carriers combined) ───────────────────
with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, ax = plt.subplots(figsize=(15, 4))

    lc_specs = [
        (lcoh2_instantaneous, lcoh2_mwh, r"LCOH$_2$", "tab:green"),
        (lcoel_instantaneous,  lcoel_mwh,  "LCOEl",   "tab:olive"),
        (lcoh_instantaneous,  lcoh_mwh,  "LCOH",      "tab:orange"),
        (lcoe_instantaneous,  lcoe_mwh,  "LCOE",      "purple"),
    ]

    for lc_ts, annual_lc, label, color in lc_specs:
        sorted_vals = lc_ts.dropna().sort_values(ascending=False).values
        ax.plot(sorted_vals, color=color, lw=1.2, label=f"{label} duration curve | average: {annual_lc:.1f} €/MWh")
        #ax.plot([0, len(sorted_vals) - 1], [annual_lc, annual_lc], color=color, lw=1, ls="--", label=f"{label} annual avg: {annual_lc:.1f} €/MWh")

    ax.set_xlabel("Hours (ranked)")
    ax.set_ylabel("Levelized Cost \n [€/MWh]")
    ax.legend(fontsize=20)

    plt.tight_layout()
    func.savefigure(fig, "MODEL_lc_duration_curves")
    #plt.show()





# =============================================================================
# PLOT 9 — Replacement cost breakdown for key components
# =============================================================================

PROJECT_N = par.LIFETIME  # 20 years

# ── PV ────────────────────────────────────────────────────────────────────────
# Panels   : permanent (n=20, no replacement)
# Inverter : replaced every 15 years over 20-year project
pv_panels_bottom              = func.cost_permanent(par.PV_CAPEX_PANELS,   invest_sizes["pv"], PROJECT_N, par.WACC)
pv_inverter_first, pv_inverter_repl = func.split_cost(par.PV_CAPEX_INVERTER, invest_sizes["pv"], PROJECT_N, par.WACC, u=par.PV_LIFETIME_INVERTER)

pv_bottom = pv_panels_bottom + pv_inverter_first
pv_top    = pv_inverter_repl

# ── Electrolyser ──────────────────────────────────────────────────────────────
# System : permanent (n=20, no replacement)
# Stack  : replaced every 5 years → 3 replacements over 20 years
el_system_bottom              = func.cost_permanent(par.ELECTROLYSER_CAPEX_SYSTEM, invest_sizes["electrolyser"], PROJECT_N, par.WACC)
el_stack_first, el_stack_repl = func.split_cost(par.ELECTROLYSER_CAPEX_STACK, invest_sizes["electrolyser"], PROJECT_N, par.WACC, u=par.ELECTROLYSER_LIFETIME_STACK)

el_bottom = el_system_bottom + el_stack_first
el_top    = el_stack_repl

# ── Fuel Cell ─────────────────────────────────────────────────────────────────
# System : permanent (n=20, no replacement)
# Stack  : replaced every 5 years → 3 replacements over 20 years
fc_system_bottom              = func.cost_permanent(par.FUELCELL_CAPEX_SYSTEM, invest_sizes["fuelcell"], PROJECT_N, par.WACC)
fc_stack_first, fc_stack_repl = func.split_cost(par.FUELCELL_CAPEX_STACK, invest_sizes["fuelcell"], PROJECT_N, par.WACC, u=par.FUELCELL_LIFETIME_STACK)

fc_bottom = fc_system_bottom + fc_stack_first
fc_top    = fc_stack_repl

# ── Battery ───────────────────────────────────────────────────────────────────
# System BOS : replaced every 10 years → 1 replacement over 20 years
# Pack       : replaced every  5 years → 3 replacements over 20 years
bat_system_first, bat_system_repl = func.split_cost(par.BATTERY_CAPEX_SYSTEM,       invest_sizes["battery_energy"], PROJECT_N, par.WACC, u=par.BATTERY_LIFETIME_SYSTEM)
bat_pack_first,   bat_pack_repl   = func.split_cost(par.BATTERY_CAPEX_BATTERY_PACK, invest_sizes["battery_energy"], PROJECT_N, par.WACC, u=par.BATTERY_LIFETIME_BATTERY_PACK)

bat_bottom = bat_system_first + bat_pack_first
bat_top    = bat_system_repl  + bat_pack_repl

# ── Salt Cavern ───────────────────────────────────────────────────────────────
# Geological + cushion gas : permanent (n=20, no replacement)  [€/MWh × MWh]
# Compressor               : replaced every 15 years           [€/MW  × MW]
sc_geo_bottom                   = func.cost_permanent(par.SALTCAVERN_CAPEX + par.SALTCAVERN_CAPEX_CUSHION_GAS, invest_sizes["saltcavern_energy"], PROJECT_N, par.WACC)
sc_comp_first, sc_comp_repl     = func.split_cost(par.SALTCAVERN_CAPEX_COMPRESSOR, invest_sizes["saltcavern_power"], PROJECT_N, par.WACC, u=par.SALTCAVERN_LIFETIME_COMPRESSOR)

sc_bottom = sc_geo_bottom + sc_comp_first
sc_top    = sc_comp_repl

# ── Collect for plotting ──────────────────────────────────────────────────────
labels     = ["PV", "Electrolyser", "Fuel Cell", "Battery", "Salt Cavern"]
cost_bottom = [pv_bottom, el_bottom, fc_bottom, bat_bottom, sc_bottom]
cost_top    = [pv_top,    el_top,    fc_top,    bat_top,    sc_top]

x     = np.arange(len(labels))
width = 0.5


with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, ax = plt.subplots(figsize=(6, 4))

    # ── Stacked bar ───────────────────────────────────────────────────────────
    cb = [v / 1e6 for v in cost_bottom]
    ct = [v / 1e6 for v in cost_top]

    ax.bar(x, cb, width, label="Up to 1st replacement", color="steelblue")
    ax.bar(x, ct, width, label="Replacements",          color="firebrick", bottom=cb)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Lifetime Costs / [M€]")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # ── Secondary y-axis: replacement % as line plot ──────────────────────────
    pct = [t / b * 100 if b > 0 else 0.0 for t, b in zip(cost_top, cost_bottom)]

    ax2 = ax.twinx()
    ax2.plot(x, pct, color="black", lw=1.5, ls="--", marker="o", markersize=5, zorder=5, label="Repl. / Initial [%]")
    ax2.set_ylabel("Replacement / Initial cost [%]")
    ax2.tick_params(axis="y")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    # ── Combined legend ───────────────────────────────────────────────────────
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=9, loc="upper left")

    func.savefigure(fig, "MODEL_replacement_cost_overhead")
    #plt.show()





# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print(f"  {'Component':<15} {'Initial (M€/yr)':>16} {'Replacements (M€/yr)':>21} {'Total (M€/yr)':>14} {'Repl/Init %':>12}")
print("=" * 85)
for i, name in enumerate(labels):
    total = cost_bottom[i] + cost_top[i]
    print(
        f"  {name:<15}"
        f"  {cost_bottom[i]/1e6:>14.3f}"
        f"  {cost_top[i]/1e6:>19.3f}"
        f"  {total/1e6:>12.3f}"
        f"  {pct[i]:>10.1f}%"
    )
print("=" * 85)


# =============================================================================
# FINANCIAL METRICS — NPV, IRR, Payback, H2 Selling Price
# =============================================================================

# ── selling prices ──────────────────────────────────────────────────────────
MARGIN           = par.MARGIN
H2_SELLING_PRICE = lcoh2_mwh / (1 - MARGIN)               # €/MWh_H2
ELECTRICITY_SELLING_PRICE = lcoel_mwh / (1 - MARGIN)        # €/MWh_el
HEAT_SELLING_PRICE        = lcoh_mwh / (1 - MARGIN)        # €/MWh_th

# ── Annual revenue (H2 sales only) ────────────────────────────────────────────
annual_revenue_h2 = total_h2_mwh * H2_SELLING_PRICE           # €/yr
annual_revenue_el = total_el_mwh * ELECTRICITY_SELLING_PRICE  # €/yr
annual_revenue_th = total_heat_mwh * HEAT_SELLING_PRICE      # €/yr

annual_revenue = annual_revenue_h2 + annual_revenue_el + annual_revenue_th  # €/yr total revenue from all carriers

# ── Annual variable OPEX ──────────────────────────────────────────────────────
annual_var_opex = sum(variable_costs_total.values())        # €/yr

# ── Annual fixed OPEX (capacity-based, excluded from annuity for financial model)
annual_fixed_opex = (
      invest_sizes["pv"]                    * par.PV_OPEX
    + invest_sizes["wind"]                  * par.WIND_OPEX
    + invest_sizes["electrolyser"]          * par.ELECTROLYSER_OPEX
    + invest_sizes["heatpump"]              * par.HEATPUMP_OPEX_FIX
    + invest_sizes["fuelcell"]                   * par.FUELCELL_OPEX_FIX
    + invest_sizes["battery_power"]         * par.BATTERY_OPEX_FIX
    + invest_sizes["saltcavern_energy"]     * par.SALTCAVERN_OPEX
    + invest_sizes["thermalstorage_energy"] * par.THERMALSTORAGE_OPEX
)

annual_cost_total = annual_fixed_opex + annual_var_opex     # €/yr
annual_cashflow   = annual_revenue - annual_cost_total      # €/yr net of OPEX

# ── Total fixed cost (annualised CAPEX only, no OPEX) — used for reference ───
total_fixed_cost = sum(ep_costs_map[k] * invest_sizes[k] for k in ep_costs_map)  # €/yr

# ── Total upfront CAPEX (undiscounted, year-0 investment) ─────────────────────
capex_total = (
      invest_sizes["pv"]                * (par.PV_CAPEX_PANELS + par.PV_CAPEX_INVERTER)
    + invest_sizes["wind"]              * par.WIND_CAPEX
    + invest_sizes["electrolyser"]      * par.ELECTROLYSER_CAPEX
    + invest_sizes["heatpump"]          * par.HEATPUMP_CAPEX
    + invest_sizes["fuelcell"]               * par.FUELCELL_CAPEX    
    + invest_sizes["battery_energy"]    * par.BATTERY_CAPEX
    + invest_sizes["battery_power"]     * par.BATTERY_CAPEX_POWER
    + invest_sizes["saltcavern_energy"] * (par.SALTCAVERN_CAPEX + par.SALTCAVERN_CAPEX_CUSHION_GAS)
    + invest_sizes["saltcavern_power"]  * par.SALTCAVERN_CAPEX_COMPRESSOR
    + invest_sizes["thermalstorage_energy"] * par.THERMALSTORAGE_CAPEX
)
print(f"Total CAPEX (undiscounted)        : {capex_total:,.0f} €")
print(f"Total annualised fixed costs      : {total_fixed_cost:,.0f} €/yr")

# ── Project horizon & cash flow array ─────────────────────────────────────────
horizon   = par.LIFETIME
cashflows = np.array([-capex_total] + [annual_cashflow] * horizon)

# ── NPV timeseries ────────────────────────────────────────────────────────────
years      = np.arange(0, horizon + 1)
npv_series = np.array([npf.npv(par.WACC, cashflows[:t + 1]) for t in range(len(cashflows))])

# ── Scalar metrics ────────────────────────────────────────────────────────────
npv_final = npf.npv(par.WACC, cashflows)
irr       = npf.irr(cashflows)

# ── Discounted payback (interpolated zero-crossing of NPV) ────────────────────
payback_idx_disc = np.where(npv_series >= 0)[0]
if len(payback_idx_disc) > 0:
    t0_d = payback_idx_disc[0] - 1
    t1_d = payback_idx_disc[0]
    frac_disc          = -npv_series[t0_d] / (npv_series[t1_d] - npv_series[t0_d])
    payback_discounted = t0_d + frac_disc
else:
    payback_discounted = None

print("\n" + "=" * 52)
print("  FINANCIAL METRICS")
print("=" * 52)
print(f"  LCOH2              : {lcoh2_mwh:>12.2f} €/MWh_H2  ({lcoh2_kg:.2f} €/kg)")
print(f"  LCOEl               : {lcoel_mwh:>12.2f} €/MWh_el")
print(f"  LCOH               : {lcoh_mwh:>12.2f} €/MWh_th")
print(f"  LCOE (system)      : {lcoe_mwh:>12.2f} €/MWh  (all carriers)")
print(f"  Margin             : {MARGIN*100:>10.1f} %")
print(f"  H2 selling price   : {H2_SELLING_PRICE:>10.2f} €/MWh_H2")
print(f"  El selling price   : {ELECTRICITY_SELLING_PRICE:>10.2f} €/MWh_el")
print(f"  Heat selling price : {HEAT_SELLING_PRICE:>10.2f} €/MWh_th")
print(f"  H2 Profit per MWh  : {H2_SELLING_PRICE - lcoh2_mwh:>10.2f} €/MWh_H2")
print(f"  El Profit per MWh  : {ELECTRICITY_SELLING_PRICE - lcoel_mwh:>10.2f} €/MWh_el")
print(f"  Heat Profit per MWh: {HEAT_SELLING_PRICE - lcoh_mwh:>10.2f} €/MWh_th")
print(f"  Annual revenue     : {annual_revenue:>12.0f} €/yr")
print(f"  Annual OPEX total  : {annual_cost_total:>12.0f} €/yr")
print(f"  Annual variable OPEX: {annual_var_opex:>12.0f} €/yr")
print(f"  Annual fixed OPEX  : {annual_fixed_opex:>12.0f} €/yr")
print(f"  Annual net cashflow: {annual_cashflow:>12.0f} €/yr")
print(f"  Total CAPEX        : {capex_total:>12.0f} €")
print(f"  Project horizon    : {horizon:>10} years")
print(f"  NPV ({horizon} yr) : {npv_final/1e6:>10.2f} M€")
print(f"  IRR                : {irr*100 if irr is not None else float('nan'):>10.2f} %")
if payback_discounted is not None:
    print(f"  Payback (discounted): {payback_discounted:>9.2f} years")
else:
    print(f"  Payback (discounted):   >{horizon} years (not recovered)")
print("=" * 52)


# =============================================================================
# PLOT 10 — NPV evolution over project lifetime
# =============================================================================

with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(years, npv_series / 1e6, color="steelblue", linewidth=1.5,
            linestyle="--", marker="o", markersize=3, zorder=3, label="NPV")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    if payback_discounted is not None:
        ax.axvline(payback_discounted, color="black", linewidth=0.8, linestyle="--",
                   label=f"Payback Time = {payback_discounted:.1f} yrs")
        ax.plot(payback_discounted, 0, "o", color="black", markersize=5, zorder=4)

    ax.set_xlabel("Year")
    ax.set_ylabel("NPV / [M€]")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    func.savefigure(fig, "MODEL_npv_evolution")
    #plt.show()


#THIS IS THE SAME PLOT AS ABOVE, BUT WITH CASHFLOWS
'''# ── Discounted cash flows per year ────────────────────────────────────────────
disc_factors   = np.array([(1 + par.WACC) ** t for t in years])
disc_cashflows = cashflows / disc_factors          # element-wise discounting

# year 0 = CAPEX (negative), years 1..N = net annual cashflow (positive)
disc_capex    = disc_cashflows[0]                  # scalar, negative
disc_annual   = disc_cashflows[1:]                 # array, positive

with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax_npv = ax.twinx()

    # ── Bars: discounted cash flow per year ───────────────────────────────────
    ax.bar(0, disc_capex / 1e6,
           color="firebrick", alpha=0.8, label="CAPEX (discounted)")
    ax.bar(years[1:], disc_annual / 1e6,
           color="steelblue", alpha=0.8, label="Net cashflow (discounted)")
    ax.axhline(0, color="black", lw=0.8, ls="--")

    # ── NPV cumulative line on secondary y-axis ───────────────────────────────
    ax_npv.plot(years, npv_series / 1e6,
                color="black", lw=1.5, ls="--", marker="o", markersize=3,
                zorder=5, label="Cumulative NPV")
    ax_npv.axhline(0, color="black", lw=0.5, ls=":")

    if payback_discounted is not None:
        ax_npv.axvline(payback_discounted, color="black", lw=0.8, ls=":",
                       label=f"Payback = {payback_discounted:.1f} yrs")
        ax_npv.plot(payback_discounted, 0, "o", color="black", markersize=5, zorder=6)

    ax_npv.set_ylabel("Cumulative NPV / [M€]")

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlabel("Year")
    ax.set_ylabel("Discounted Cash Flow / [M€]")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_npv.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=10)

    plt.tight_layout()
    func.savefigure(fig, "MODEL_npv1_evolution")
    plt.show()'''


# =============================================================================
# FLEXIBILITY METRICS CALCULATION
# =============================================================================


# ── 1. Residual load on every bus ───────────────────────────────────
res_load_el = func.residual_load(
    consumption = el_demand_ts + el_input + heatpump_input, #NOTE: No battery charging here, as that is a flexible load that can be shifted in time. We want to see the residual load before any flexibility is applied.
    generation = wind_flow + pv_flow + fuelcell_output_el)

res_load_h2 = func.residual_load(
    consumption = h2_demand_ts + fuelcell_input, 
    generation = el_output)

res_load_heat = func.residual_load(
    consumption = heat_demand_ts, 
    generation = heatpump_output + fuelcell_output_heat + el_output_heat)


res_load = res_load_el + res_load_h2 + res_load_heat




# ── 2. Storage metrics ────────────────────────────────────────────────────────
flex_battery = {
    "utilisation_rate":    func.storage_utilisation_rate(battery_soc, capacity_battery),
    "throughput_cycles_in":   func.throughput_ratio(bat_charge, capacity_battery),
    "throughput_cycles_out":    func.throughput_ratio(bat_discharge, capacity_battery),
    "load_shift_index": func.load_shift_index(
        discharge         = bat_discharge,
        soc               = battery_soc,
        capacity          = capacity_battery,
        power_rating      = invest_sizes["battery_power"],
        min_soc_fraction  = 0.0,
        dt                = 1.0,
    ),
    **func.storage_flexibility_band(battery_soc, capacity_battery, 
                                    invest_sizes["battery_power"], 0),
}

flex_saltcavern = {
    "utilisation_rate":    func.storage_utilisation_rate(saltcavern_soc, capacity_saltcavern),
    "throughput_cycles_in":   func.throughput_ratio(saltcavern_charge, capacity_saltcavern),
    "throughput_cycles_out":    func.throughput_ratio(saltcavern_discharge, capacity_saltcavern),
    "load_shift_index": func.load_shift_index(
        discharge         = saltcavern_discharge,
        soc               = saltcavern_soc,
        capacity          = capacity_saltcavern,
        power_rating      = invest_sizes["saltcavern_power"],
        min_soc_fraction  = par.SALTCAVERN_CUSHIONGAS_FRACTION,
        dt                = 1.0,
    ),
    **func.storage_flexibility_band(saltcavern_soc,capacity_saltcavern,
                                    invest_sizes["saltcavern_power"], par.SALTCAVERN_CUSHIONGAS_FRACTION),
}

flex_thermalstorage = {
    "utilisation_rate":    func.storage_utilisation_rate(thermalstorage_soc, capacity_thermalstorage),
    "throughput_cycles_in":   func.throughput_ratio(thermalstorage_charge, capacity_thermalstorage),
    "throughput_cycles_out":    func.throughput_ratio(thermalstorage_discharge, capacity_thermalstorage),
    "load_shift_index":    func.load_shift_index(
        discharge         = thermalstorage_discharge,
        soc               = thermalstorage_soc,
        capacity          = capacity_thermalstorage,
        power_rating      = invest_sizes["thermalstorage_power"],
        min_soc_fraction  = 0.0,
        dt                = 1.0,
    ),
    **func.storage_flexibility_band(thermalstorage_soc, capacity_thermalstorage, 
                                    invest_sizes["thermalstorage_power"], 0.0),
}

# ── 3. Sector-coupling metrics ────────────────────────────────────────────────
total_el_produced = pv_flow.sum() + wind_flow.sum() + fuelcell_output_el.sum()

flex_electrolyser = func.electrolyser_flexibility_metrics(
    el_input        = el_input,
    el_output_h2    = el_output,
    el_output_heat  = el_output_heat,
    capacity        = capacity_electrolyser,
    total_el_mwh    = total_el_produced,
    total_h2_mwh    = h2_demand_ts.sum(),
    efficiency_h2   = par.ELECTROLYSER_EFFICIENCY,
    efficiency_heat = par.ELECTROLYSER_RECOVERABLE_HEAT,
)

flex_fuelcell = func.chp_flexibility_metrics(
    h2_input    = fuelcell_input,
    el_output   = fuelcell_output_el,
    heat_output = fuelcell_output_heat,
    capacity    = capacity_fuelcell,
    total_h2_mwh = h2_demand_ts.sum(),
)

flex_heatpump = {
    "capacity_factor": float(heatpump_input.mean() / capacity_heatpump) if capacity_heatpump > 0 else 0.0,
    "sc_ratio_el_to_heat": func.sector_coupling_ratio(heatpump_input.sum(), total_el_produced),
    "ramp_up_avg_mw_per_h": float(heatpump_input.diff().clip(lower=0).mean()),
    "ramp_down_avg_mw_per_h": float(heatpump_input.diff().clip(upper=0).abs().mean()),
    "hours_operating": int((heatpump_input > 0).sum()),
}

# ── 4. Demand-side / curtailment metrics ──────────────────────────────────────
flex_curtailment = {
    **func.curtailment_metrics(curtailment_el,   total_el_produced,   "electricity"),
    **func.curtailment_metrics(curtailment_h2,   el_output.sum(),  "h2"),
    **func.curtailment_metrics(curtailment_heat, el_output_heat.sum() + heatpump_output.sum() + fuelcell_output_heat.sum(),"heat"),
}

# ── 5. Ramping metrics ────────────────────────────────────────────────────────
flex_ramping = func.ramping_metrics(res_load_el)

# ── 6. System-level ratios ────────────────────────────────────────────────────
total_local_gen_el = pv_flow + wind_flow + fuelcell_output_el
flex_system = {
    "ssr_electricity":  func.self_sufficiency_ratio(el_demand_ts, total_local_gen_el),
    "scr_electricity":  func.self_consumption_ratio(el_demand_ts, pv_flow + wind_flow),
    "sc_ratio_el_to_h2_lossless":    func.sector_coupling_ratio(el_input.sum(), total_el_produced),
    "sc_ratio_el_to_h2_with_losses": func.sector_coupling_ratio(el_input.sum(), total_el_produced,
                                        efficiency=par.ELECTROLYSER_EFFICIENCY, with_losses=True),
    "sc_ratio_el_to_heat_lossless":  func.sector_coupling_ratio(heatpump_input.sum(), total_el_produced),
    "sc_ratio_h2_to_el_and_heat":    func.sector_coupling_ratio(fuelcell_input.sum(), el_output.sum()), 
    "res_penetration":  float(total_el_produced / (el_demand_ts.sum() + el_input.sum() + heatpump_input.sum())),
}

# ── 7. Combine into a master dict and print ───────────────────────────────────
all_flex_metrics = {
    "battery":        flex_battery,
    "salt_cavern":    flex_saltcavern,
    "thermal_storage":flex_thermalstorage,
    "electrolyser":   flex_electrolyser,
    "fuelcell":            flex_fuelcell,
    "heat_pump":      flex_heatpump,
    "curtailment":    flex_curtailment,
    "ramping":        flex_ramping,
    "system":         flex_system,
}

flex_table = func.flexibility_summary_table(all_flex_metrics)

print("\n" + "=" * 70)
print("  FLEXIBILITY METRICS SUMMARY")
print("=" * 70)
for cat, metrics in all_flex_metrics.items():
    print(f"\n  [{cat.upper()}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<40} {v:>10.4f}")
        else:
            print(f"    {k:<40} {v:>10}")
print("=" * 70)




# =============================================================================
# PLOT F1 — Normalised SoC for all three storages (full year)
# Purpose : Shows when each storage technology is providing flexibility
#           and identifies seasonal patterns in storage cycling.
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for ax, soc, cap, label, color, flex_dict in zip(
    axes,
    [battery_soc,      saltcavern_soc,      thermalstorage_soc],
    [capacity_battery, capacity_saltcavern,  capacity_thermalstorage],
    ["Battery",        "Salt Cavern",        "Thermal Storage"],
    ["purple",         "brown",              "orange"],
    [flex_battery,     flex_saltcavern,      flex_thermalstorage],
):
    soc_norm = soc / cap if cap > 0 else soc * 0
    ax.fill_between(soc.index, soc_norm, alpha=0.4, color=color)
    #ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("SoC / capacity [-]")
    ax.set_ylim(0, 1.05)
    ur = flex_dict["utilisation_rate"]
    capacity = cap
    li = flex_dict["throughput_cycles_out"]
    tc = flex_dict["throughput_cycles_in"]
    ax.set_title(
        f"{label}   |   Utilisation rate: {ur:.2%}      Capacity: {capacity:.0f} MWh "
        f"Total Energy Disharged: {li:.1f} cycles/year   Total Energy Charged: {tc:.1f} cycles/yr",
        fontsize=9,
    )

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].set_xlabel("Month")
plt.suptitle("Normalised State of Charge — All Storage Technologies", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F1_soc_normalised_all_storages")
#plt.show()

# =============================================================================
# PLOT F2 — Residual load overview (single panel)
# Bottom x-axis : calendar time  → deficit / surplus fill_between time series
# Top x-axis    : fraction of year (duration curve axis)
#                 → total residual load duration curve (black dashed)
#                 → per-bus residual load duration curves (coloured)
# =============================================================================


with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, ax_left = plt.subplots(figsize=(14, 5))

    ax_left.fill_between(
        res_load.index,
        np.where(res_load > 0, res_load.values, 0),
        alpha=0.5, color="steelblue", zorder=1, label="Deficit",
    )
    ax_left.fill_between(
        res_load.index,
        np.where(res_load < 0, res_load.values, 0),
        alpha=0.5, color="firebrick", zorder=1, label="Surplus",
    )

    ax_left.set_ylabel("Residual Load / [MW]")
    ax_left.set_xlabel("Month")
    ax_left.xaxis.set_major_locator(mdates.MonthLocator())
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax_dur = ax_left.twiny()
    #ax_dur.set_xlim(0, 1) #NOTE: This is what removes margins from plots
    ax_dur.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    n_hours = len(res_load)
    x_norm  = np.linspace(0, 1, n_hours)

    rl_sorted = res_load.sort_values(ascending=False).values
    ax_dur.plot(x_norm, rl_sorted,
                color="black", lw=1.6, zorder=3, label="System total")

    bus_specs = [
        (res_load_el,   "Electricity bus", "tab:olive"),
        (res_load_h2,   "Hydrogen bus",    "tab:green"),
        (res_load_heat, "Heat bus",        "tab:orange"),
    ]

    for z, (rl_ts, label, color) in enumerate(bus_specs, start=4):
        sorted_rl = rl_ts.sort_values(ascending=False).values
        ax_dur.plot(x_norm, sorted_rl, color=color, lw=1.5, zorder=z, label=label)

    handles_ts,  labels_ts  = ax_left.get_legend_handles_labels()
    handles_dur, labels_dur = ax_dur.get_legend_handles_labels()
    ax_left.legend(
        handles_ts + handles_dur,
        labels_ts  + labels_dur,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,        # override the rc_context frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])   # leave room on the right for legend

    plt.tight_layout()
    func.savefigure(fig, "FLEX_F2_residual_load_overview")
    #plt.show()

# ── FLEX PLOT 2 — Ramping distribution (violin + box, all buses) ──────────────
# The ramp distribution reveals the "speed" of flexibility needed.
# Wide distributions = high ramping requirements → need fast-response assets.


'''Might be useful:
# ── Centre: 1-hour ramp distribution ─────────────────────────────────────────
ramp_1h_ts = res_load.diff(1).dropna()
axes[1].hist(ramp_1h_ts, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
axes[1].axvline(
    flex_ramping["ramp_1h_max_mw"], color="red", ls="--", lw=1.2,
    label=f"Max ramp: {flex_ramping['ramp_1h_max_mw']:.1f} MW",
)
axes[1].axvline(
    flex_ramping["ramp_1h_p95_mw"], color="orange", ls="--", lw=1.2,
    label=f"P95 ramp: {flex_ramping['ramp_1h_p95_mw']:.1f} MW",
)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("1-hour ramp [MWh/h]")
axes[1].set_ylabel("Hours")
axes[1].set_title(
    f"1-hour Δ(Residual Load) Distribution\n"
    f"Std. Dev.: {flex_ramping['ramp_std_mw']:.1f} MW   "
    f"Max: {flex_ramping['ramp_1h_max_mw']:.1f} MW"
)
axes[1].legend(fontsize=8)'''
'''fig, axes = plt.subplots(1, 1, figsize=(16, 5))

for ax, rl_ts, label in zip(
    axes,
    [res_load_el, res_load_h2, res_load_heat],
    ["Electricity", "Hydrogen", "Heat"],
):
    ramp_data = {}
    for w in label:
        ramp_data[f"{w}h label"] = rl_ts.diff(w).values  #This is absolute change in net load over the window, not normalised by time, so units are MW/h
        #To get the average ramp rate in MW/h, we could divide by the window size (w), but the absolute change is more intuitive for understanding the magnitude of flexibility needed.

    df_ramp = pd.DataFrame(ramp_data)
    df_ramp_melt = df_ramp.melt(var_name="Window", value_name="Ramp [MW/h]")

    sns.violinplot(data=df_ramp_melt, x="Window", y="Ramp [MW/h]",
                   ax=ax, hue="Window", palette="Set2", cut=0, inner="box")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"{label} Bus\nRamping Distribution")

plt.suptitle("Ramping Flexibility Requirements (1h / 3h / 8h windows)",
             fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F2b_residual_load_distributions")
plt.show()'''


fig, ax = plt.subplots(1, 1, figsize=(16, 5))

ramp_data = {}
for label, rl_ts in zip(["Electricity", "Hydrogen", "Heat"], [res_load_el, res_load_h2, res_load_heat]):
    ramp_data[label] = rl_ts.diff(1).values  # 1-hour ramp

df_ramp = pd.DataFrame(ramp_data)
df_ramp_melt = df_ramp.melt(var_name="Bus", value_name="Hourly Load Variation [MW]")

sns.violinplot(data=df_ramp_melt, x="Bus", y="Hourly Load Variation [MW]",
                ax=ax, hue="Bus", palette="Set2", cut=0, inner="box")
#ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Ramping Flexibility Requirements (1h Ramps for All Buses)")

plt.tight_layout()
func.savefigure(fig, "FLEX_F2b_residual_load_distributions")
#plt.show()




# =============================================================================
# PLOT F4 — Monthly flexibility contribution by source
# Purpose : Breaks down each month's total flexibility provision into its
#           contributing sources. Shows seasonality and how the balance
#           between storage, sector coupling and curtailment shifts.
#
# Flexibility provision (MWh) per source per month:
#   - Battery discharge
#   - Salt cavern discharge
#   - Thermal storage discharge
#   - Electrolyser (absorbing surplus → enabling H2 storage)
#   - Heat pump (absorbing surplus → heat)
#   - fuelcell (converting H2 back to electricity when needed)
#
# Flexibility need (MWh) per month:
#   - Curtailment (electricity, H2, heat) — unserved flexibility
# =============================================================================

monthly_flex = pd.DataFrame({
    "Battery discharge":         bat_discharge.values,
    "Salt cavern discharge":     saltcavern_discharge.values,
    "Thermal storage discharge": thermalstorage_discharge.values,
    "Electrolyser (P2H2)":       el_input.values,
    "Heat pump (P2H)":           heatpump_input.values,
    "Fuel Cell (H2 → El+Heat)":  fuelcell_input.values,
    "Curtailment El":            curtailment_el.values,
    "Curtailment H2":            curtailment_h2.values,
    "Curtailment Heat":          curtailment_heat.values,
}, index=idx).resample("ME").sum()

supply_cols = [
    "Battery discharge",
    "Salt cavern discharge",
    "Thermal storage discharge",
    "Electrolyser (P2H2)",
    "Heat pump (P2H)",
    "Fuel Cell (H2 → El+Heat)",
]
failure_cols = ["Curtailment El", "Curtailment H2", "Curtailment Heat"]

supply_colors = ["purple", "brown", "orange", "tab:blue", "tab:red", "tab:green"]
failure_colors = ["firebrick", "darkred", "tomato"]

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# ── Top: flexibility provision ────────────────────────────────────────────────
bottoms = np.zeros(len(monthly_flex))
x = np.arange(len(monthly_flex))
for col, color in zip(supply_cols, supply_colors):
    vals = monthly_flex[col].values / 1e3  # GWh
    axes[0].bar(x, vals, bottom=bottoms / 1e3, label=col, color=color, width=0.8)
    bottoms += monthly_flex[col].values

axes[0].set_ylabel("Energy Flow [GWh/month]")
axes[0].set_title("Energy Delievered to Destination Bus")
axes[0].legend(loc="upper right", fontsize=8, ncol=2)

# ── Bottom: curtailment (flexibility failure) ─────────────────────────────────
bottoms = np.zeros(len(monthly_flex))
for col, color in zip(failure_cols, failure_colors):
    vals = monthly_flex[col].values / 1e3  # GWh
    axes[1].bar(x, vals, bottom=bottoms / 1e3, label=col, color=color, width=0.8, alpha=0.8)
    bottoms += monthly_flex[col].values

axes[1].set_ylabel("Curtailment [GWh/month]")
axes[1].set_title("Monthly Lost Potential (Curtailment) by Carrier")
axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [d.strftime("%b") for d in monthly_flex.index], rotation=45
)
axes[1].legend(loc="upper right", fontsize=8)

plt.suptitle("Monthly Flexibility: Provision vs Failure", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F4_monthly_flexibility_provision_vs_failure")
#plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 4b: Weekly rolling flexibility intensity
# Purpose : Shows how the balance between storage-based and sector-coupling
#           flexibility shifts over the year. A 7-day rolling mean smooths
#           out hourly noise while preserving seasonal trends.
# ──────────────────────────────────────────────────────────────────────────────
weekly_storage_flex = (
    bat_discharge + saltcavern_discharge + thermalstorage_discharge
).rolling(24 * 7, min_periods=1).mean()

weekly_sc_flex = (
    el_input + heatpump_input + fuelcell_input
).rolling(24 * 7, min_periods=1).mean()

weekly_curtailment = (
    curtailment_el + curtailment_h2 + curtailment_heat
).rolling(24 * 7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(weekly_storage_flex.index, weekly_storage_flex.values,
                alpha=0.5, color="purple", label="Storage-based flows (7-day avg)")
ax.fill_between(weekly_sc_flex.index,
                weekly_storage_flex.values,
                weekly_storage_flex.values + weekly_sc_flex.values,
                alpha=0.5, color="tab:blue", label="Sector-coupling flows (7-day avg)")
ax.plot(weekly_curtailment.index, weekly_curtailment.values,
        color="firebrick", lw=1.5, ls="--", label="Curtailment (7-day avg)")

ax.set_ylabel("MW (7-day rolling mean)")
ax.set_xlabel("Month")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Rolling Weekly Flexibility Intensity\n"
             "Storage-based vs Sector-coupling vs Curtailment")

plt.tight_layout()
func.savefigure(fig, "FLEX_F4b_rolling_flexibility_intensity")
#plt.show()



# ── FLEX PLOT 4c — Rolling flexibility index (weekly rolling window) ────────────
# Shows how the flexibility burden shifts throughout the year.
# Computed as the fraction of electricity demand met by non-RES sources
# (storage + fuelcell), rolling 7-day window.
#TODO: think about flex_index_ variables' defintion. DO they make sense? Do they capture the intended concept of "flexibility reliance"?
window = 7*24  # 7 days × 24 h

non_res_el = bat_discharge + fuelcell_output_el  # non-RES contributions to el bus
flex_index_el = (non_res_el / el_demand_ts.replace(0, np.nan)).rolling(window).mean()

non_res_h2    = saltcavern_discharge  # non-electrolyser H2
flex_index_h2 = (non_res_h2 / h2_demand_ts.replace(0, np.nan)).rolling(window).mean()

non_res_heat  = thermalstorage_discharge + el_output_heat
flex_index_heat = (non_res_heat / heat_demand_ts.replace(0, np.nan)).rolling(window).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(flex_index_el.index,   flex_index_el.values,   label="Electricity bus",  color="tab:olive",  lw=1.2)
ax.plot(flex_index_h2.index,   flex_index_h2.values,   label="Hydrogen bus",     color="tab:green",   lw=1.2)
ax.plot(flex_index_heat.index, flex_index_heat.values, label="Heat bus",         color="tab:orange",   lw=1.2)
ax.axhline(1, color="black", lw=0.5, ls="--")
ax.set_ylabel("Fraction of demand met by non-primary sources,\n 7d average")
ax.set_xlabel("Date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend()
plt.tight_layout()
func.savefigure(fig, "FLEX_F4c_rolling_fraction_of_demand_met_by_non_primary_sources")
#plt.show()


# =============================================================================
# PLOT F7 — Sector Coupling: supply fraction and absolute energy
# =============================================================================
# 4 bars: El→H2 | El→Heat | H2→El | H2→Heat
#
# Colour logic — one colour family per technology:
#   Electrolyser  → blue family  (H2 output: steelblue, waste heat: lightsteelblue)
#   Heat Pump     → green family (heat output: seagreen)
#   fuelcell           → red family   (electricity: firebrick, heat: lightcoral)
#
# Bar 1 "El→H2"   : electrolyser H2 output only          → steelblue
# Bar 2 "El→Heat" : HP output (seagreen) STACKED ON TOP
#                   electrolyser waste heat (lightsteelblue)
# Bar 3 "H2→El"   : fuelcell electricity output only          → firebrick
# Bar 4 "H2→Heat" : fuelcell heat output only                 → lightcoral

# ── Colour palette ────────────────────────────────────────────────────────────
C_ELEC    = "steelblue"
C_HP      = "seagreen"
C_fuelcell     = "firebrick"

# ── Supply denominators ───────────────────────────────────────────────────────
total_res_gen = pv_flow.sum() + wind_flow.sum()
total_h2_gen  = el_output.sum()

# ── Individual component values ───────────────────────────────────────────────
ratio_elec_h2    = el_output.sum()       / total_res_gen * 100 if total_res_gen > 0 else 0.0
energy_elec_h2   = float(el_output.sum() / 1e3)

ratio_elec_waste = el_output_heat.sum()  / total_res_gen * 100 if total_res_gen > 0 else 0.0
ratio_hp         = heatpump_output.sum() / total_res_gen * 100 if total_res_gen > 0 else 0.0
energy_elec_waste= float(el_output_heat.sum()  / 1e3)
energy_hp        = float(heatpump_output.sum() / 1e3)

ratio_fuelcell_el     = fuelcell_output_el.sum()   / total_h2_gen * 100 if total_h2_gen > 0 else 0.0
energy_fuelcell_el    = float(fuelcell_output_el.sum()   / 1e3)

ratio_fuelcell_heat   = fuelcell_output_heat.sum() / total_h2_gen * 100 if total_h2_gen > 0 else 0.0
energy_fuelcell_heat  = float(fuelcell_output_heat.sum() / 1e3)

# ── Bar positions ─────────────────────────────────────────────────────────────
x_pos      = np.array([0, 1, 2, 3])
bar_width  = 0.5
bar_labels = [
    "El → H₂\n(Electrolyser)",
    "El → Heat\n(Electrolyser waste\n+ HP)",
    "H₂ → El\n(Fuel Cell)",
    "H₂ → Heat\n(Fuel Cell)",
]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# ── Bars on ax1 (GWh, left axis) ─────────────────────────────────────────────

# Bar 1
ax1.bar(x_pos[0], energy_elec_h2, bar_width,
        color=C_ELEC, alpha=0.88, edgecolor="white",
        label="Electrolyser → H₂")

# Bar 2: stacked
ax1.bar(x_pos[1], energy_elec_waste, bar_width,
        color=C_ELEC, alpha=0.88, edgecolor="white",
        label="Electrolyser → Waste heat")
ax1.bar(x_pos[1], energy_hp, bar_width,
        bottom=energy_elec_waste,
        color=C_HP, alpha=0.88, edgecolor="white",
        label="Heat Pump → Heat")

# Bar 3
ax1.bar(x_pos[2], energy_fuelcell_el, bar_width,
        color=C_fuelcell, alpha=0.88, edgecolor="white",
        label="Fuel Cell → Electricity")

# Bar 4
ax1.bar(x_pos[3], energy_fuelcell_heat, bar_width,
        color=C_fuelcell, alpha=0.88, edgecolor="white",
        label="Fuel Cell → Heat")

# ── Y-axis limits ─────────────────────────────────────────────────────────────
max_energy = max(energy_elec_h2,
                 energy_elec_waste + energy_hp,
                 energy_fuelcell_el,
                 energy_fuelcell_heat)
max_ratio  = max(ratio_elec_h2,
                 ratio_elec_waste + ratio_hp,
                 ratio_fuelcell_el,
                 ratio_fuelcell_heat)

ax1.set_ylim(0, max_energy * 1.22)
ax2.set_ylim(0, max_ratio  * 1.22)   # same headroom → bars align visually

# ── Top annotations: "X.X GWh (Y.Y%)" ───────────────────────────────────────
func.annotate_top(ax1, x_pos[0], energy_elec_h2,                unit=f" GWh ({ratio_elec_h2:.1f}%)")
func.annotate_top(ax1, x_pos[1], energy_elec_waste + energy_hp, unit=f" GWh ({ratio_elec_waste + ratio_hp:.1f}%)")
func.annotate_top(ax1, x_pos[2], energy_fuelcell_el,                 unit=f" GWh ({ratio_fuelcell_el:.1f}%)")
func.annotate_top(ax1, x_pos[3], energy_fuelcell_heat,               unit=f" GWh ({ratio_fuelcell_heat:.1f}%)")

# ── Inner annotations for stacked bar 2 ──────────────────────────────────────
func.annotate_inner(ax1, x_pos[1], energy_elec_waste / 2,              f"Waste\n{energy_elec_waste:.1f} GWh")
func.annotate_inner(ax1, x_pos[1], energy_elec_waste + energy_hp / 2,  f"HP\n{energy_hp:.1f} GWh")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax1.set_xticks(x_pos)
ax1.set_xticklabels(bar_labels, fontsize=9)
ax1.set_ylabel("Annual energy output crossing boundary [GWh]")
ax2.set_ylabel("% of Energy Carrier Supply Crossing Between Buses")
ax2.axhline(100, color="grey", lw=0.6, ls="--", alpha=0.4)
ax1.set_xlabel("Bus Pairs and Responsible Components")

ax1.legend(fontsize=8, loc="upper right")
ax1.set_title(
    "Absolute Cross-Sector Energy Flows and Corresponding Supply Fraction\n"
    "       Bars 1-2: % of RES electricity and absolute values\n" 
    "       Bars 3-4: % of H₂ produced and absolute values",
    fontsize=9,
)

'''plt.suptitle(
    "Sector Coupling — Supply Fraction and Absolute Energy Crossing Sector Boundaries\n " \
    "Sector Coupling Ratio\n"
    "Bars 1-2: % of RES electricity and absolute values\n" 
    "Bars 3-4: % of H₂ produced and absolute values")'''
plt.tight_layout()
func.savefigure(fig, "FLEX_F7_sector_coupling_ratios")
#plt.show()




# =============================================================================
# PLOT F8 — Quantitative flexibility comparison: radar chart
#         — Two radar charts: Storage flexibility | PtX flexibility 
# Purpose : Provides a single visual that compares all flexibility sources
#           on a common normalised scale. Each axis is one normalised metric.
#           A larger area = more flexibility contribution from that source.

# Axes (6 total):
#   1. Utilisation          — storage SoC avg OR capacity factor for converters
#   2. Load-shift index     — energy cycled / capacity (storage only)
#   3. Upward flexibility   — avg upward ramp / capacity
#   4. Downward flexibility — avg downward ramp / capacity
#   5. Responsiveness       — fraction of online hours with active ramping
#   6. PtX up/down flex     — for converters: headroom/footroom relative to capacity
#                             (replaces throughput cycles and SC ratio)
# =============================================================================

# Normalise each metric to [0,1] relative to its system maximum
# so that all axes are directly comparable on the radar.


# ── Storage metrics ────────────────────────────────────────────────────────────
n_hours = len(bat_charge)


bat_resp = func.responsiveness(bat_discharge + bat_charge,                   capacity_battery)
sc_resp  = func.responsiveness(saltcavern_discharge + saltcavern_charge,     capacity_saltcavern)
ts_resp  = func.responsiveness(thermalstorage_discharge + thermalstorage_charge, capacity_thermalstorage)

bat_utilisation = (battery_soc.mean()        / capacity_battery)        if capacity_battery        > 0 else 0.0
sc_utilisation  = (saltcavern_soc.mean()     / capacity_saltcavern)     if capacity_saltcavern     > 0 else 0.0
ts_utilisation  = (thermalstorage_soc.mean() / capacity_thermalstorage) if capacity_thermalstorage > 0 else 0.0

bat_up  = func.safeval(flex_battery["upward_mwh"])        / (capacity_battery        ) if capacity_battery        > 0 else 0.0
sc_up   = func.safeval(flex_saltcavern["upward_mwh"])     / (capacity_saltcavern     ) if capacity_saltcavern     > 0 else 0.0
ts_up   = func.safeval(flex_thermalstorage["upward_mwh"]) / (capacity_thermalstorage ) if capacity_thermalstorage  > 0 else 0.0

bat_dn  = func.safeval(flex_battery["downward_mwh"])        / (capacity_battery       ) if capacity_battery       > 0 else 0.0
sc_dn   = func.safeval(flex_saltcavern["downward_mwh"])     / (capacity_saltcavern    ) if capacity_saltcavern    > 0 else 0.0
ts_dn   = func.safeval(flex_thermalstorage["downward_mwh"]) / (capacity_thermalstorage) if capacity_thermalstorage > 0 else 0.0

bat_ssr = func.self_sufficiency_ratio(el_demand_ts, bat_discharge)
sc_ssr  = func.self_sufficiency_ratio(h2_demand_ts, saltcavern_discharge)
ts_ssr  = func.self_sufficiency_ratio(heat_demand_ts, thermalstorage_discharge)

bat_lsi = flex_battery["load_shift_index"] 
sc_lsi = flex_saltcavern["load_shift_index"]
ts_lsi = flex_thermalstorage["load_shift_index"]

# ── PtX metrics ────────────────────────────────────────────────────────────────
el_resp  = func.responsiveness(el_input,       capacity_electrolyser)
hp_resp  = func.responsiveness(heatpump_input, capacity_heatpump)
fuelcell_resp = func.responsiveness(fuelcell_input,      capacity_fuelcell)

#CFs already defined in cf_electrolyser, cf_heatpump, cf_fuelcell

el_up  = func.ptx_upward_flex(el_input,       capacity_electrolyser)
hp_up  = func.ptx_upward_flex(heatpump_input, capacity_heatpump)
fuelcell_up = func.ptx_upward_flex(fuelcell_input,      capacity_fuelcell)

el_dn  = func.ptx_downward_flex(el_input,       capacity_electrolyser, par.ELECTROLYSER_MIN_LOAD)
hp_dn  = func.ptx_downward_flex(heatpump_input, capacity_heatpump,     par.HEATPUMP_MIN_LOAD)
fuelcell_dn = func.ptx_downward_flex(fuelcell_input,      capacity_fuelcell,          par.FUELCELL_MIN_LOAD)

'''el_ramp_util  = func.ramp_utilisation(el_input,       capacity_electrolyser, par.ELECTROLYSER_MIN_LOAD)
hp_ramp_util  = func.ramp_utilisation(heatpump_input, capacity_heatpump,     par.HEATPUMP_MIN_LOAD)
fuelcell_ramp_util = func.ramp_utilisation(fuelcell_input,      capacity_fuelcell,          par.FUELCELL_MIN_LOAD)'''

#NOTE:SCR and SSR calculated differently for electrolyser and fuelcell
#     Because these units have 2 ouptuts each

el_total_output = el_output.sum() + el_output_heat.sum()
frac_h2_elec    = el_output.sum()      / el_total_output
frac_heat_elec  = el_output_heat.sum() / el_total_output

ssr_elec_h2   = func.self_sufficiency_ratio(h2_demand_ts,   el_output)
ssr_elec_heat = func.self_sufficiency_ratio(heat_demand_ts, el_output_heat)
scr_elec_h2   = func.self_consumption_ratio(h2_demand_ts,   el_output)
scr_elec_heat = func.self_consumption_ratio(heat_demand_ts, el_output_heat)


fuelcell_total_output = fuelcell_output_el.sum() + fuelcell_output_heat.sum()
frac_el_fuelcell   = fuelcell_output_el.sum()   / fuelcell_total_output
frac_heat_fuelcell = fuelcell_output_heat.sum() / fuelcell_total_output

ssr_fuelcell_el   = func.self_sufficiency_ratio(el_demand_ts,   fuelcell_output_el)
ssr_fuelcell_heat = func.self_sufficiency_ratio(heat_demand_ts, fuelcell_output_heat)
scr_fuelcell_el   = func.self_consumption_ratio(el_demand_ts,   fuelcell_output_el)
scr_fuelcell_heat = func.self_consumption_ratio(heat_demand_ts, fuelcell_output_heat)


el_ssr        = frac_h2_elec * ssr_elec_h2 + frac_heat_elec * ssr_elec_heat
hp_ssr        = func.self_sufficiency_ratio(heat_demand_ts, heatpump_output)
fuelcell_ssr       = frac_el_fuelcell * ssr_fuelcell_el + frac_heat_fuelcell * ssr_fuelcell_heat


el_scr        = frac_h2_elec * scr_elec_h2 + frac_heat_elec * scr_elec_heat
hp_scr        = func.self_consumption_ratio(heat_demand_ts, heatpump_output)
fuelcell_scr       = frac_el_fuelcell * scr_fuelcell_el + frac_heat_fuelcell * scr_fuelcell_heat
# ── Radar data ─────────────────────────────────────────────────────────────────
''

storage_radar = {
    "Battery": {
        "Responsiveness":         bat_resp,
        "Utilisation\n(SOC avg)": bat_utilisation,
        "Upward\nflexibility":    bat_up,
        "Downward\nflexibility":  bat_dn,
        "SSR":                    bat_ssr,
        "Output\npotential":      bat_lsi,        
    },
    "Salt Cavern": {
        "Responsiveness":         sc_resp,
        "Utilisation\n(SOC avg)": sc_utilisation,
        "Upward\nflexibility":    sc_up,
        "Downward\nflexibility":  sc_dn,
        "SSR":                    sc_ssr,
        "Output\npotential":      sc_lsi,  
    },}
    
'''  
    "Thermal Storage": {
        "Responsiveness":         ts_resp,
        "Utilisation\n(SoC avg)": ts_utilisation,
        "Upward\nflexibility":    ts_up,
        "Downward\nflexibility":  ts_dn,
        "SSR":                    ts_ssr,
        "Output\npotential":      ts_lsi,
    },
'''



ptx_radar = {
    "Electrolyser": {
        "Responsiveness":            el_resp,
        "Capacity\nfactor":          cf_electrolyser,
        "Upward\nflexibility":       el_up,
        "Downward\nflexibility":     el_dn,
        "SSR":                       el_ssr,
        "SCR":                       el_scr, 
    },
    "Heat Pump": {
        "Responsiveness":            hp_resp,
        "Capacity\nfactor":          cf_heatpump,
        "Upward\nflexibility":       hp_up,
        "Downward\nflexibility":     hp_dn,
        "SSR":                       hp_ssr,
        "SCR":                       hp_scr,  
    },
    "Fuel Cell": {
        "Responsiveness":            fuelcell_resp,
        "Capacity\nfactor":          cf_fuelcell,
        "Upward\nflexibility":       fuelcell_up,
        "Downward\nflexibility":     fuelcell_dn,
        "SSR":                       fuelcell_ssr,
        "SCR":                       fuelcell_scr,
    },
}

with mpl.rc_context({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.format": "pdf",
}):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=True))

    func.draw_radar(axes[0], storage_radar,
                    colors=["purple", "brown", "orange"],
                    abc="(a)")

    func.draw_radar(axes[1], ptx_radar,
                    colors=["tab:blue", "tab:green", "tab:red"],
                    abc="(b)")
    
    
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    func.savefigure(fig, "MODEL_radar_flexibility")
    #plt.show()

# ── Console summary ────────────────────────────────────────────────────────────
for label, data in [("STORAGE", storage_radar), ("PtX", ptx_radar)]:
    cats = list(list(data.values())[0].keys())
    print("\n" + "=" * 160)
    print(f"  {label} RADAR — RAW VALUES")
    print("=" * 160)
    print(f"  {'Technology':<18}", end="")
    for c in cats:
        print(f"  {c.replace(chr(10), ' '):<22}", end="")
    print()
    print("  " + "-" * 158)
    for tech, vals in data.items():
        print(f"  {tech:<18}", end="")
        for c in cats:
            print(f"  {vals[c]:>22.3f}", end="")
        print()
    print("=" * 160)





# =============================================================================
# FLEXIBILITY SUMMARY TABLE — print and save
# =============================================================================

print("\n" + "=" * 75)
print("  FLEXIBILITY METRICS — FULL QUANTITATIVE SUMMARY")
print("=" * 75)
print(f"\n  {'Category':<20} {'Metric':<40} {'Value':>10}")
print(f"  {'-'*72}")

for cat, metrics in all_flex_metrics.items():
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {cat:<20} {k:<40} {v:>10.4f}")
        elif isinstance(v, int):
            print(f"  {cat:<20} {k:<40} {v:>10d}")

print("=" * 75)

# =============================================================================
# LOGBOOK — CSV
# =============================================================================

logbook = {

    "scenario": SCENARIO_NAME,

    # ── Key model parameters ──────────────────────────────────────────────────
    "parameters": {
        "wacc":                              par.WACC,

        "general_lifetime":                  par.LIFETIME,

        "margin":                            par.MARGIN,

        "pv_lifetime_system_yr":             par.PV_LIFETIME_SYSTEM,
        "pv_lifetime_inverter_yr":           par.PV_LIFETIME_INVERTER,

        "wind_lifetime_yr":                  par.WIND_LIFETIME_SYSTEM,

        "electrolyser_efficiency":           par.ELECTROLYSER_EFFICIENCY,
        "electrolyser_efficiency_heat":      par.ELECTROLYSER_RECOVERABLE_HEAT,
        "electrolyser_min_load":             par.ELECTROLYSER_MIN_LOAD,
        "electrolyser_lifetime_system":      par.ELECTROLYSER_LIFETIME_SYSTEM,
        "electrolyser_lifetime_stack":       par.ELECTROLYSER_LIFETIME_STACK,

        "heatpump_cop":                      par.HEATPUMP_COP,
        "heatpump_min_load":                 par.HEATPUMP_MIN_LOAD,
        "heatpump_lifetime_yr":              par.HEATPUMP_LIFETIME,

        "fuelcell_efficiency_el":                 par.FUELCELL_EFFICIENCY_ELECTRICITY,
        "fuelcell_efficiency_heat":               par.FUELCELL_EFFICIENCY_HEAT,
        "fuelcell_min_load":                      par.FUELCELL_MIN_LOAD,
        "fuelcell_lifetime_system":               par.FUELCELL_LIFETIME_SYSTEM,
        "fuelcell_lifetime_stack":                par.FUELCELL_LIFETIME_STACK,

        "battery_efficiency":                par.BATTERY_EFFICIENCY_RT,
        "battery_self_discharge_rate":       par.BATTERY_SELF_DISCHARGE_RATE,
        "battery_lifetime_system":           par.BATTERY_LIFETIME_SYSTEM,
        "battery_lifetime_pack":             par.BATTERY_LIFETIME_BATTERY_PACK,
        "battery_min_charge_time_h":         par.BATTERY_MIN_CHARGE_TIME,

        "saltcavern_efficiency":             par.SALTCAVERN_EFFICIENCY,
        "saltcavern_self_discharge_rate":    par.SALTCAVERN_SELF_DISCHARGE_RATE,
        "saltcavern_lifetime_system":        par.SALTCAVERN_LIFETIME,
        "saltcavern_lifetime_compressor":    par.SALTCAVERN_LIFETIME_COMPRESSOR,
        "saltcavern_min_soc":                par.SALTCAVERN_CUSHIONGAS_FRACTION,

        "thermalstorage_efficiency":         par.THERMALSTORAGE_EFFICIENCY_RT,
        "thermalstorage_self_discharge_rate":par.THERMALSTORAGE_SELF_DISCHARGE_RATE,
        "thermalstorage_lifetime_system":    par.THERMALSTORAGE_LIFETIME,
    },

    # ── Optimal capacities ────────────────────────────────────────────────────
    "capacities": {
        "capacity_pv_mw":                    float(capacity_pv),

        "capacity_wind_mw":                  float(capacity_wind),

        "capacity_total_renewable_mw":       float(capacity_pv + capacity_wind),

        "capacity_electrolyser_mw":          float(capacity_electrolyser),

        "capacity_fuelcell_mw":              float(capacity_fuelcell),

        "capacity_heatpump_mw":              float(capacity_heatpump),

        "capacity_battery_energy_mwh":       float(capacity_battery),
        "capacity_battery_power_mw":         float(invest_sizes["battery_power"]),

        "capacity_saltcavern_energy_mwh":    float(capacity_saltcavern),
        "capacity_saltcavern_compressor_mw": float(invest_sizes["saltcavern_power"]),

        "capacity_thermalstorage_mwh":       float(capacity_thermalstorage),
        "capacity_thermalstorage_power_mw":  float(invest_sizes["thermalstorage_power"]),
    },

    # ── Operational metrics ───────────────────────────────────────────────────
    "operational": {
        "capacity_factor_pv":                func.capacity_factor(pv_flow,   capacity_pv,   dt) if capacity_pv   > 0 else 0.0,
        "capacity_factor_wind":              func.capacity_factor(wind_flow, capacity_wind, dt) if capacity_wind > 0 else 0.0,
        "capacity_factor_res_avg":           func.capacity_factor(pv_flow + wind_flow, capacity_pv + capacity_wind, dt) if (capacity_pv + capacity_wind) > 0 else 0.0,
        "capacity_factor_electrolyser":      func.capacity_factor(el_input, capacity_electrolyser, dt) if capacity_electrolyser > 0 else 0.0,
        "electrolyser_frac_h2":              float(electro_frac_h2),
        "electrolyser_frac_heat":            float(electro_frac_heat),
        "capacity_factor_fuelcell":          func.capacity_factor(fuelcell_input, capacity_fuelcell, dt) if capacity_fuelcell > 0 else 0.0,
        "fuelcell_frac_el":                  float(fuelcell_frac_el),
        "fuelcell_frac_heat":                float(fuelcell_frac_heat),
        "capacity_factor_heatpump":          func.capacity_factor(heatpump_input, capacity_heatpump, dt) if capacity_heatpump > 0 else 0.0,
        "total_h2_demanded_mwh":             float(total_h2_mwh),
        "total_h2_demanded_kg":              float(total_h2_kg),
        "total_heat_demanded_mwh":           float(total_heat_mwh),
        "total_el_served_mwh":               float(el_demand_ts.sum()),
        "total_renewable_gen_mwh":           float(pv_flow.sum() + wind_flow.sum()),
        "curtailment_el_mwh":                float(curtailment_el.sum()),
        "curtailment_h2_mwh":                float(curtailment_h2.sum()),
        "curtailment_heat_mwh":              float(curtailment_heat.sum()),
        "curtailment_el_fraction":           float(curtailment_el.sum() / (pv_flow.sum() + wind_flow.sum())) if (pv_flow.sum() + wind_flow.sum()) > 0 else 0.0,
        "curtailment_h2_fraction_of_demand": float(curtailment_h2.sum() / total_h2_mwh) if total_h2_mwh > 0 else 0.0,
        "curtailment_heat_fraction":         float(curtailment_heat.sum() / total_heat_mwh) if total_heat_mwh > 0 else 0.0,
        "curtailment_total_fraction":        float(curtailment_el.sum() + curtailment_h2.sum() + curtailment_heat.sum()) / (pv_flow.sum() + wind_flow.sum()) if (pv_flow.sum() + wind_flow.sum()) > 0 else 0.0,
        "curtailment_share":                 float(par.CURTAILMENT_SHARE), #should have the exact same value as the variable above
        "capacity_factor_battery_energy":    func.capacity_factor(battery_soc,   invest_sizes["battery_energy"], dt) if invest_sizes["battery_energy"] > 0 else 0.0,
        "capacity_factor_battery_power":     func.capacity_factor(bat_discharge, invest_sizes["battery_power"],  dt) if invest_sizes["battery_power"]  > 0 else 0.0,
        "battery_charge_mwh":                float(bat_charge.sum()),
        "battery_discharge_mwh":             float(bat_discharge.sum()),
        "capacity_factor_saltcavern_energy": func.capacity_factor(saltcavern_soc,       invest_sizes["saltcavern_energy"], dt) if invest_sizes["saltcavern_energy"] > 0 else 0.0,
        "capacity_factor_saltcavern_power":  func.capacity_factor(saltcavern_discharge, invest_sizes["saltcavern_power"],  dt) if invest_sizes["saltcavern_power"]  > 0 else 0.0,
        "saltcavern_charge_mwh":             float(saltcavern_charge.sum()),
        "saltcavern_discharge_mwh":          float(saltcavern_discharge.sum()),
        "capacity_factor_thermalstorage_energy": func.capacity_factor(thermalstorage_soc,       invest_sizes["thermalstorage_energy"], dt) if invest_sizes["thermalstorage_energy"] > 0 else 0.0,
        "capacity_factor_thermalstorage_power":  func.capacity_factor(thermalstorage_discharge, invest_sizes["thermalstorage_power"], dt) if invest_sizes["thermalstorage_power"] > 0 else 0.0,
        "thermalstorage_charge_mwh":         float(thermalstorage_charge.sum()),
        "thermalstorage_discharge_mwh":      float(thermalstorage_discharge.sum()),
    },

    # ── levelized costs ───────────────────────────────────────────────────────
    "levelized_cost": {
        "lcoh2_eur_per_mwh":   float(lcoh2_mwh),
        "lcoh2_eur_per_kg":    float(lcoh2_kg),
        "lcoel_eur_per_mwh":    float(lcoel_mwh),
        "lcoh_eur_per_mwh":    float(lcoh_mwh),
        "lcoe_eur_per_mwh":    float(lcoe_mwh),
    },

    # ── Commercial ────────────────────────────────────────────────────────────
    "commercial": {
        "margin_applied":                        float(MARGIN),
        "h2_selling_price_eur_per_mwh":          float(H2_SELLING_PRICE),
        "h2_profit_eur_per_mwh":                 float(H2_SELLING_PRICE - lcoh2_mwh),
        "electricity_selling_price_eur_per_mwh": float(ELECTRICITY_SELLING_PRICE),
        "electricity_profit_eur_per_mwh":        float(ELECTRICITY_SELLING_PRICE - lcoel_mwh),
        "heat_selling_price_eur_per_mwh":        float(HEAT_SELLING_PRICE),
        "heat_profit_eur_per_mwh":               float(HEAT_SELLING_PRICE - lcoh_mwh),
    },

    # ── CAPEX undiscounted ────────────────────────────────────────────────────
    "capex_undiscounted": {
        "pv_panels_eur":             float(invest_sizes["pv"]               * par.PV_CAPEX_PANELS),
        "pv_inverter_eur":           float(invest_sizes["pv"]               * par.PV_CAPEX_INVERTER),
        "pv_total_eur":              float(invest_sizes["pv"]               * (par.PV_CAPEX_PANELS + par.PV_CAPEX_INVERTER)),

        "wind_eur":                  float(invest_sizes["wind"]             * par.WIND_CAPEX),

        "electrolyser_system_eur":   float(invest_sizes["electrolyser"]     * par.ELECTROLYSER_CAPEX_SYSTEM),
        "electrolyser_stack_eur":    float(invest_sizes["electrolyser"]     * par.ELECTROLYSER_CAPEX_STACK),
        "electrolyser_total_eur":    float(invest_sizes["electrolyser"]     * par.ELECTROLYSER_CAPEX),

        "heatpump_eur":              float(invest_sizes["heatpump"]          * par.HEATPUMP_CAPEX),

        "fuelcell_system_eur":            float(invest_sizes["fuelcell"]               * par.FUELCELL_CAPEX_SYSTEM),
        "fuelcell_stack_eur":             float(invest_sizes["fuelcell"]               * par.FUELCELL_CAPEX_STACK),
        "fuelcell_total_eur":             float(invest_sizes["fuelcell"]               * par.FUELCELL_CAPEX),

        "battery_system_eur":        float(invest_sizes["battery_energy"]   * par.BATTERY_CAPEX_SYSTEM),
        "battery_pack_eur":          float(invest_sizes["battery_energy"]   * par.BATTERY_CAPEX_BATTERY_PACK),
        "battery_total_eur":         float(invest_sizes["battery_energy"]   * par.BATTERY_CAPEX),

        "saltcavern_geological_eur": float(invest_sizes["saltcavern_energy"]* par.SALTCAVERN_CAPEX),
        "saltcavern_cushiongas_eur": float(invest_sizes["saltcavern_energy"]* par.SALTCAVERN_CAPEX_CUSHION_GAS),
        "saltcavern_compressor_eur": float(invest_sizes["saltcavern_power"] * par.SALTCAVERN_CAPEX_COMPRESSOR),

        "thermalstorage_eur":        float(invest_sizes["thermalstorage_energy"] * par.THERMALSTORAGE_CAPEX),

        "total_eur":                 float(capex_total),
    },

    # ── Annualised CAPEX (ep_costs × size, €/yr) ──────────────────────────────
    "capex_annualized_eur_per_yr": {
        k: float(ep_costs_map[k] * invest_sizes[k]) for k in ep_costs_map
    },

    # ── Fixed OPEX (€/yr) ─────────────────────────────────────────────────────
    "opex_fixed_eur_per_yr": {
        "pv":             float(invest_sizes["pv"]               * par.PV_OPEX),
        "wind":           float(invest_sizes["wind"]             * par.WIND_OPEX),
        "electrolyser":   float(invest_sizes["electrolyser"]     * par.ELECTROLYSER_OPEX),
        "heatpump":       float(invest_sizes["heatpump"]          * par.HEATPUMP_OPEX_FIX),
        "fuelcell":       float(invest_sizes["fuelcell"]               * par.FUELCELL_OPEX_FIX),
        "battery":        float(invest_sizes["battery_power"]    * par.BATTERY_OPEX_FIX),
        "saltcavern":     float(invest_sizes["saltcavern_energy"]* par.SALTCAVERN_OPEX),
        "thermalstorage": float(invest_sizes["thermalstorage_energy"] * par.THERMALSTORAGE_OPEX),
        "total":          float(annual_fixed_opex),
    },

    # ── Variable OPEX (€/yr) ──────────────────────────────────────────────────
    "opex_variable_eur_per_yr": {
        "heatpump_variable":   float(variable_costs_total["heatpump_variable"]),
        "battery_charge":      float(variable_costs_total["battery_charge"]),
        "battery_discharge":   float(variable_costs_total["battery_discharge"]),
        "saltcavern_compress": float(variable_costs_total["saltcavern_compress"]),
        "total":               float(annual_var_opex),
    },

    # ── Annual cashflow model ─────────────────────────────────────────────────
    "cashflow": {
        "annual_fixed_opex_eur":        float(annual_fixed_opex),
        "annual_variable_opex_eur":     float(annual_var_opex),
        "annual_total_opex_eur":        float(annual_cost_total), # = annual_fixed_opex + annual_var_opex
        "annual_revenue_eur":           float(annual_revenue),
        "annual_net_cashflow_eur":      float(annual_cashflow), # = annual_revenue - annual_cost_total
        "total_capex_undiscounted_eur": float(capex_total),
    },

    # ── Financial metrics ─────────────────────────────────────────────────────
    "financial": {
        "total_system_cost_eur_per_yr": float(total_cost), # = model.objective()
        "npv_final_eur":                float(npv_final),
        "irr":                          float(irr)                if irr                is not None else None,
        "payback_discounted_yr":        float(payback_discounted) if payback_discounted is not None else None,
        "project_horizon_yr":           int(horizon),
    },
}

# With this:
logbook["flexibility_summary"] = {
    f"{cat}.{metric}": val
    for cat, metrics in all_flex_metrics.items()
    for metric, val in metrics.items()
}

# ── Radar-ready normalised flexibility metrics [0,1] ─────────────────────────
logbook["flexibility_radar"] = {
    # Storage
    "battery.responsiveness":    bat_resp,
    "battery.utilisation":       bat_utilisation,
    "battery.upward_flex":       bat_up,
    "battery.downward_flex":     bat_dn,
    "battery.ssr":               bat_ssr,
    "battery.load_shift_index":  bat_lsi,

    "salt_cavern.responsiveness":   sc_resp,
    "salt_cavern.utilisation":      sc_utilisation,
    "salt_cavern.upward_flex":      sc_up,
    "salt_cavern.downward_flex":    sc_dn,
    "salt_cavern.ssr":              sc_ssr,
    "salt_cavern.load_shift_index": sc_lsi,

    "thermal_storage.responsiveness":   ts_resp,
    "thermal_storage.utilisation":      ts_utilisation,
    "thermal_storage.upward_flex":      ts_up,
    "thermal_storage.downward_flex":    ts_dn,
    "thermal_storage.ssr":              ts_ssr,
    "thermal_storage.load_shift_index": ts_lsi,

    # PtX
    "electrolyser.responsiveness":  el_resp,
    "electrolyser.capacity_factor": cf_electrolyser,
    "electrolyser.upward_flex":     el_up,
    "electrolyser.downward_flex":   el_dn,
    "electrolyser.ssr":             el_ssr,
    "electrolyser.scr":             el_scr,

    "heat_pump.responsiveness":  hp_resp,
    "heat_pump.capacity_factor": cf_heatpump,
    "heat_pump.upward_flex":     hp_up,
    "heat_pump.downward_flex":   hp_dn,
    "heat_pump.ssr":             hp_ssr,
    "heat_pump.scr":             hp_scr,

    "fuelcell.responsiveness":  fuelcell_resp,
    "fuelcell.capacity_factor": cf_fuelcell,
    "fuelcell.upward_flex":     fuelcell_up,
    "fuelcell.downward_flex":   fuelcell_dn,
    "fuelcell.ssr":             fuelcell_ssr,
    "fuelcell.scr":             fuelcell_scr,
}

func.save_logbook_csv(logbook, DATA_DIR, SCENARIO_NAME)