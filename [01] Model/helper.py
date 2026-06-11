'''Author: Gonçalo Costa Pina
Date_Created: 2026-01-30 (30th January 2026)
Date_Modified: 2026-04-18

----------------------------------------------

Defines the main working script for the project, which:
- Sets up and solves the oemof optimization model
- Produces visualisations of the main results

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

# =============================================================================
# SCENARIO DEFINITION
# =============================================================================
SCENARIO_NAME = "FINAL_MODEL"   # ← change per run


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
            minimum=0, existing=0,
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
            minimum=0, 
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
            minimum=0, 
            existing=0,
        ),
        min=par.HEATPUMP_MIN_LOAD,
        variable_costs=par.HEATPUMP_OPEX_VAR,
    )},
    outputs={bus_heat: solph.Flow()},
    conversion_factors={bus_heat: cop_ts_list},
)

# ── CHP (fuel cell) ───────────────────────────────────────────────────────────
chp = solph.components.Converter(
    label="chp",
    inputs={bus_H2: solph.Flow(
        nominal_value=solph.Investment(
            ep_costs=(
                economics.annuity(capex=par.FUELCELL_CAPEX_SYSTEM, n=par.FUELCELL_LIFETIME_SYSTEM, wacc=par.WACC)
                + economics.annuity(capex=par.FUELCELL_CAPEX_STACK,  n=par.FUELCELL_LIFETIME_SYSTEM, wacc=par.WACC, u=par.FUELCELL_LIFETIME_STACK)
                + par.FUELCELL_OPEX_FIX
            ),
            minimum=0, 
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
    invest_relation_output_capacity=1 / par.BATTERY_MIN_DISCHARGE_TIME,
    inflow_conversion_factor=par.BATTERY_EFFICIENCY_CHARGE,
    outflow_conversion_factor=par.BATTERY_EFFICIENCY_DISCHARGE,
    loss_rate=par.BATTERY_SELF_DISCHARGE_RATE,
    initial_storage_level=0,
)

# ── Salt Cavern ───────────────────────────────────────────────────────────────
saltcavern = solph.components.GenericStorage( # TODO: shrinkage rate
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
    invest_relation_input_capacity=None,
    invest_relation_output_capacity=None,
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
    invest_relation_input_capacity=None,
    invest_relation_output_capacity=None,
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
    electrolyser, chp, heatpump,
    battery, saltcavern, thermalstorage,
    ElectricityDemand, H2Demand, HeatDemand,
    electricity_curtailement, H2_curtailement, heat_curtailement,
)

model = solph.Model(energysystem)

print("Running optimization...\n")
model.solve(solver='cbc', solve_kwargs={'tee': True})
print("Processing results...\n")

results        = solph.processing.results(model)
string_results = views.convert_keys_to_strings(results)

# ── Optimal capacities ────────────────────────────────────────────────────────
capacity_pv             = results[(pv, bus_electricity)]['scalars']['invest']
capacity_wind           = results[(wind, bus_electricity)]['scalars']['invest']
capacity_electrolyser   = results[(bus_electricity, electrolyser)]['scalars']['invest']
capacity_chp            = results[(bus_H2, chp)]['scalars']['invest']
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
print(f"  CHP:                     {capacity_chp:.1f} MW")
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

chp_input        = results[(bus_H2, chp)]['sequences']['flow']
chp_output_el    = results[(chp, bus_electricity)]['sequences']['flow']
chp_output_heat  = results[(chp, bus_heat)]['sequences']['flow']

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
chp_input        = chp_input.iloc[:-1]
chp_output_el    = chp_output_el.iloc[:-1]
chp_output_heat  = chp_output_heat.iloc[:-1]
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


# =============================================================================
# PLOT 1 — Monthly energy flows
# =============================================================================

monthly = pd.DataFrame({
    "PV":                    pv_flow.values,
    "Wind":                  wind_flow.values,
    "Electrolyser":          el_input.values,
    "Heat Pump":             heatpump_input.values,
    "CHP":                   chp_input.values,
    "Electricity Curtailment": curtailment_el.values,
    "H2 Curtailment":        curtailment_h2.values,
    "Heat Curtailment":      curtailment_heat.values,
    "Electricity Demand":    el_demand_ts.values,
    "H2 Demand":             h2_demand_ts.values,
    "Heat Demand":           heat_demand_ts.values,
}, index=idx).resample("ME").sum()

fig, ax = plt.subplots(figsize=(14, 5))
monthly[["PV", "Wind", "Electrolyser", "Heat Pump", "CHP",
         "Electricity Curtailment", "H2 Curtailment", "Heat Curtailment"]].plot(
    kind="bar", ax=ax, width=0.8
)
ax.set_ylabel("Energy / [MWh]")
ax.set_xlabel("")
ax.set_xticklabels([d.strftime("%b") for d in monthly.index], rotation=45)
ax.legend(loc="best")
plt.tight_layout()
func.savefigure(fig, "MODEL_monthly_flows")
plt.show()


# =============================================================================
# PLOT 2 — Sample week: hourly dispatch for all 3 buses
# =============================================================================

sl = slice("2023-01-10", "2023-01-17")

# ── Electricity bus ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

windplot = wind_flow.loc[sl]
pvplot   = pv_flow.loc[sl]
chpplot  = chp_output_el.loc[sl]
x        = windplot.index
base     = np.zeros_like(windplot.values)

axes[0].fill_between(x, base, base + windplot.values,                  label="Wind",                  color="tab:blue",   alpha=0.7, edgecolor='none')
base += windplot.values
axes[0].fill_between(x, base, base + pvplot.values,                    label="PV",                    color="tab:orange", alpha=0.7, edgecolor='none')
base += pvplot.values
axes[0].fill_between(x, base, base + chpplot.values,                   label="CHP electricity output", color="tab:green",  alpha=0.7, edgecolor='none')
axes[0].plot(el_input.loc[sl].index,       el_input.loc[sl].values,       label="Electrolyser input",    color="black",  lw=1.5)
axes[0].plot(heatpump_input.loc[sl].index, heatpump_input.loc[sl].values, label="Heat Pump input",       color="purple", lw=1.5)
axes[0].plot(el_demand_ts.loc[sl].index,   el_demand_ts.loc[sl].values,   label="Electricity demand",    color="orange", lw=1.5, ls="--")
axes[0].plot(curtailment_el.loc[sl].index, curtailment_el.loc[sl].values, label="Curtailment",           color="red",    lw=1,   ls="--")
axes[0].set_ylabel("MWh/h")
axes[0].legend(loc="upper right")

axes[1].plot(wind_flow.loc[sl].index, wind_flow.loc[sl].values + pv_flow.loc[sl].values + chp_output_el.loc[sl].values, label="Electricity produced", color="green",  lw=1.5)
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
plt.show()

# ── H2 bus ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

el_output_plot = el_output.loc[sl]
x    = el_output_plot.index
base = np.zeros_like(el_output_plot.values)

axes[0].fill_between(x, base, base + el_output_plot.values, label="Electrolyser Output", color="tab:blue", alpha=0.7, edgecolor='none')
axes[0].plot(chp_input.loc[sl].index,      chp_input.loc[sl].values,      label="CHP input",      color="black",  lw=1.5)
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
plt.show()

# ── Heat bus ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

electrolyser_plot = el_output_heat.loc[sl]
chpplot_heat      = chp_output_heat.loc[sl]
heatpumpplot      = heatpump_output.loc[sl]
x    = electrolyser_plot.index
base = np.zeros_like(electrolyser_plot.values)

axes[0].fill_between(x, base, base + heatpumpplot.values,                 label="Heat Pump Output",          color="tab:blue",   alpha=0.7, edgecolor='none')
base += heatpumpplot.values
axes[0].fill_between(x, base, base + chpplot_heat.values,                 label="CHP heat output",           color="tab:green",  alpha=0.7, edgecolor='none')
base += chpplot_heat.values
axes[0].fill_between(x, base, base + electrolyser_plot.values,            label="Electrolyser heat output",  color="tab:purple", alpha=0.7, edgecolor='none')
axes[0].plot(heat_demand_ts.loc[sl].index,   heat_demand_ts.loc[sl].values,   label="Heat demand",       color="orange", lw=1.5, ls="--")
axes[0].plot(curtailment_heat.loc[sl].index, curtailment_heat.loc[sl].values, label="Heat Curtailment",  color="red",    lw=1,   ls="--")
axes[0].set_ylabel("MWh/h")
axes[0].legend(loc="upper right")

axes[1].plot(heatpumpplot.index, heatpumpplot.values + chpplot_heat.values + electrolyser_plot.values, label="Heat produced", color="green",  lw=1.5)
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
plt.show()


# =============================================================================
# PLOT 3 — Full year load duration curves: Electrolyser, CHP, Heat Pump
# =============================================================================

for label, flow, capacity, fname in [
    ("Electrolyser Load / [MW]", el_input,       capacity_electrolyser, "MODEL_load_duration_curve_electrolyser"),
    ("CHP Load / [MW]",          chp_input,       capacity_chp,          "MODEL_load_duration_curve_chp"),
    ("Heat Pump Load / [MW]",    heatpump_input,  capacity_heatpump,     "MODEL_load_duration_curve_heatpump"),
]:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(flow.index, flow.values, color="red", lw=0.8)
    axes[0].axhline(capacity, color="black", ls="--", lw=1, label="Installed capacity")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel(label)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].legend()

    sorted_load = flow.sort_values(ascending=False).values
    axes[1].plot(sorted_load, color="steelblue", lw=1.5)
    axes[1].axhline(capacity, color="black", ls="--", lw=1, label="Installed capacity")
    axes[1].set_xlabel("Hours per year (ranked)")
    axes[1].set_ylabel(label)
    axes[1].legend()

    plt.tight_layout()
    func.savefigure(fig, fname)
    plt.show()


# =============================================================================
# PLOT 4 — Full year Battery SOC
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(battery_soc.index, battery_soc.values, alpha=0.5, color="purple")
ax.set_xlabel("Date")
ax.set_ylabel("Battery SOC / [MWh]")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.tight_layout()
func.savefigure(fig, "MODEL_battery_soc_year")
plt.show()


# =============================================================================
# PLOT 5 — Full year Salt Cavern SOC
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(saltcavern_soc.index, saltcavern_soc.values, alpha=0.5, color="brown")
ax.set_xlabel("Date")
ax.set_ylabel("Salt Cavern SOC / [MWh]")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.tight_layout()
func.savefigure(fig, "MODEL_saltcavern_soc_year")
plt.show()


# =============================================================================
# PLOT 6 — Full year Thermal Storage SOC
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(thermalstorage_soc.index, thermalstorage_soc.values, alpha=0.5, color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Thermal Storage SOC / [MWh]")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.tight_layout()
func.savefigure(fig, "MODEL_thermal_storage_soc_year")
plt.show()


# =============================================================================
# PLOT 7 — Levelised Cost calculations and cost attribution
# =============================================================================

dt = 1  # hours per timestep

# ── 1. Annual energy per carrier ──────────────────────────────────────────────
#TODO: check these definitions. LCOX is based on demand
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
    "chp": (
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
}

invest_sizes = {
    "pv":                    results[(pv, bus_electricity)]["scalars"]["invest"],
    "wind":                  results[(wind, bus_electricity)]["scalars"]["invest"],
    "electrolyser":          results[(bus_electricity, electrolyser)]["scalars"]["invest"],
    "heatpump":              results[(bus_electricity, heatpump)]["scalars"]["invest"],
    "chp":                   results[(bus_H2, chp)]["scalars"]["invest"],
    "battery_energy":        results[(battery, None)]["scalars"]["invest"],
    "battery_power":         results[(bus_electricity, battery)]["scalars"]["invest"],
    "saltcavern_energy":     results[(saltcavern, None)]["scalars"]["invest"],
    "saltcavern_power":      results[(bus_H2, saltcavern)]["scalars"]["invest"],
    "thermalstorage_energy": results[(thermalstorage, None)]["scalars"]["invest"],
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
cost_chp            = ep_costs_map["chp"]             * invest_sizes["chp"]
cost_battery        = (ep_costs_map["battery_energy"] * invest_sizes["battery_energy"]
                     + ep_costs_map["battery_power"]  * invest_sizes["battery_power"])
cost_saltcavern     = (ep_costs_map["saltcavern_energy"] * invest_sizes["saltcavern_energy"]
                     + ep_costs_map["saltcavern_power"]  * invest_sizes["saltcavern_power"])
cost_thermalstorage = ep_costs_map["thermalstorage_energy"] * invest_sizes["thermalstorage_energy"]

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

# ── 7. CHP cost split: electricity output vs heat output ─────────────────────
chp_total_output = chp_output_el.sum() + chp_output_heat.sum()
if chp_total_output > 0:
    chp_frac_el   = chp_output_el.sum()   / chp_total_output
    chp_frac_heat = chp_output_heat.sum() / chp_total_output
else:
    chp_frac_el   = 0.5
    chp_frac_heat = 0.5

cost_chp_el   = cost_chp  * chp_frac_el
cost_chp_heat = cost_chp  * chp_frac_heat

# ── 8. Shared supply cost (PV + Wind) allocated by energy fraction ────────────
# Method: each carrier's share of total useful energy delivered.
# Limitation: implicitly assumes 1 MWh is equally valuable across carriers.
# Alternative (input-based) is not used (for example, because CHP produces heat from H2,)
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
# Elec : supply share + battery (dedicated to electricity) + CHP electricity share
# Heat : supply share + heat pump (dedicated to heat) + thermal storage
#        + electrolyser heat fraction + CHP heat share

cost_h2_total   = (cost_supply_h2
                 + cost_electrolyser_h2
                 + cost_saltcavern
                 + cost_var_saltcavern)

cost_el_total   = (cost_supply_el
                 + cost_battery
                 + cost_var_battery
                 + cost_chp_el)

cost_heat_total = (cost_supply_heat
                 + cost_heatpump + cost_var_heatpump
                 + cost_thermalstorage
                 + cost_electrolyser_heat
                 + cost_chp_heat)

cost_system_total = cost_h2_total + cost_el_total + cost_heat_total

# ── Sanity check ──────────────────────────────────────────────────────────────
cost_manual_total = (cost_pv + cost_wind
                   + cost_electrolyser
                   + cost_heatpump + cost_var_heatpump
                   + cost_chp 
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

# ── 10. Levelised costs ───────────────────────────────────────────────────────
lcoh2_mwh = cost_h2_total   / total_h2_mwh    if total_h2_mwh   > 0 else np.nan
lcoh2_kg  = lcoh2_mwh * par.H2_CALORIFIC_VALUE_LHV
lcoe_mwh  = cost_el_total   / total_el_mwh    if total_el_mwh   > 0 else np.nan
lcoh_mwh  = cost_heat_total / total_heat_mwh  if total_heat_mwh > 0 else np.nan
lcos_mwh  = total_cost      / total_energy_mwh if total_energy_mwh > 0 else np.nan
'''NOTE: lcos_mwh could be calculated as follows. It would yield the same result:
lcos_mwh = ((lcoh2_mwh * total_h2_mwh
            + lcoe_mwh * total_el_mwh 
            + lcoh_mwh * total_heat_mwh)
            / total_energy_mwh
            ) if total_energy_mwh > 0 else np.nan'''

print("\n" + "=" * 65)
print("  LEVELISED COST RESULTS")
print("=" * 65)
print(f"  Annual H2 produced        : {total_h2_mwh:>12,.1f} MWh_H2/yr")
print(f"  Annual electricity served : {total_el_mwh:>12,.1f} MWh_el/yr")
print(f"  Annual heat produced      : {total_heat_mwh:>12,.1f} MWh_th/yr")
print(f"  Total energy              : {total_energy_mwh:>12,.1f} MWh/yr")
print(f"  Total system cost         : {total_cost:>12,.0f} €/yr")
print(f"  ---")
print(f"  LCOH2                     : {lcoh2_mwh:>12.2f} €/MWh_H2  ({lcoh2_kg:.2f} €/kg)")
print(f"  LCOE                      : {lcoe_mwh:>12.2f} €/MWh_el")
print(f"  LCOH                      : {lcoh_mwh:>12.2f} €/MWh_th")
print(f"  LCOS (system)             : {lcos_mwh:>12.2f} €/MWh  (all carriers)")
print("=" * 65)

# ── 11. Grouped cost table (fixed vs variable) ────────────────────────────────
group_map = {
    "pv":                    ("PV",              "fixed"),
    "wind":                  ("Wind",            "fixed"),
    "electrolyser":          ("Electrolyser",    "fixed"),
    "heatpump":              ("Heat Pump",       "fixed"),
    "chp":                   ("CHP",             "fixed"),
    "battery_energy":        ("Battery",         "fixed"),
    "battery_power":         ("Battery",         "fixed"),
    "saltcavern_energy":     ("Salt Cavern",     "fixed"),
    "saltcavern_power":      ("Salt Cavern",     "fixed"),
    "thermalstorage_energy": ("Thermal Storage", "fixed"),
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
def _sorted_pie(labels_vals):
    pairs = [(l, v) for l, v in labels_vals if v > 0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return zip(*pairs) if pairs else ([], [])

fixed_labels, fixed_values = _sorted_pie(
    [(tech, costs["fixed"]) for tech, costs in grouped.items()]
)
total_labels, total_values = _sorted_pie(
    [(tech, costs["fixed"] + costs["variable"]) for tech, costs in grouped.items()]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].pie(fixed_values, labels=fixed_labels, autopct='%1.1f%%', startangle=90)
axes[0].set_title("Fixed Costs Breakdown\n(Annualised CAPEX + Fixed OPEX)", fontsize=11)
axes[1].pie(total_values, labels=total_labels, autopct='%1.1f%%', startangle=90)
axes[1].set_title("Total Costs Breakdown\n(Fixed + Variable OPEX)", fontsize=11)
for ax in axes:
    ax.axis('equal')
plt.suptitle("System Cost Breakdown by Technology", fontsize=13, y=1.01)
plt.tight_layout()
func.savefigure(fig, "MODEL_cost_breakdown_pies")
plt.show()

# ── 13. Carrier cost attribution + levelised cost bar charts ──────────────────
carrier_labels = ["H2", "Electricity", "Heat"]

carrier_components = {
    "H2": {
        "Supply (shared)": cost_supply_h2,
        "Electrolyser":    cost_electrolyser_h2,
        "Salt Cavern":     cost_saltcavern + cost_var_saltcavern,
    },
    "Electricity": {
        "Supply (shared)": cost_supply_el,
        "Battery":         cost_battery + cost_var_battery,
        "CHP (el share)":  cost_chp_el,
    },
    "Heat": {
        "Supply (shared)":      cost_supply_heat,
        "Heat Pump":            cost_heatpump + cost_var_heatpump,
        "Thermal Storage":      cost_thermalstorage,
        "CHP (heat share)":     cost_chp_heat,
        "Electrolyser (heat)":  cost_electrolyser_heat,
    },
}

all_components = sorted({comp for d in carrier_components.values() for comp in d})
palette     = plt.cm.tab10.colors
comp_colors = {comp: palette[i % len(palette)] for i, comp in enumerate(all_components)}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x       = np.arange(len(carrier_labels))
bottoms = np.zeros(len(carrier_labels))

for comp in all_components:
    vals = np.array([carrier_components[carrier].get(comp, 0.0) for carrier in carrier_labels])
    axes[0].bar(x, vals / 1e6, bottom=bottoms / 1e6, label=comp, color=comp_colors[comp])
    bottoms += vals

axes[0].set_xticks(x)
axes[0].set_xticklabels(carrier_labels)
axes[0].set_ylabel("Annualised Cost / [M€/yr]")
axes[0].set_title("Attributed Cost per Energy Carrier")
axes[0].legend(loc="upper right", fontsize=8)

bar_colors = ["steelblue", "darkorange", "firebrick", "grey"]
bars = axes[1].bar(
    carrier_labels + ["System"],
    [lcoh2_mwh, lcoe_mwh, lcoh_mwh, lcos_mwh],
    color=bar_colors, width=0.5,
)
for bar, val in zip(bars, [lcoh2_mwh, lcoe_mwh, lcoh_mwh, lcos_mwh]):
    if not np.isnan(val):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9,
        )
axes[1].set_ylabel("Levelised Cost / [€/MWh]")
axes[1].set_title("Levelised Costs by Carrier\n(energy-allocation method)")
axes[1].set_ylim(0, max(lcoh2_mwh, lcoe_mwh, lcoh_mwh, lcos_mwh) * 1.15)

plt.tight_layout()
func.savefigure(fig, "MODEL_levelised_costs_by_carrier")
plt.show()

# ── 14. Cost attribution waterfall with LC annotations ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
categories = list(carrier_components.keys())
x       = np.arange(len(categories))
bottoms = np.zeros(len(categories))

for comp in all_components:
    vals = np.array([carrier_components[carrier].get(comp, 0.0) / 1e6 for carrier in categories])
    ax.bar(x, vals, bottom=bottoms, label=comp, color=comp_colors[comp])
    bottoms += vals

for i, (carrier, lc) in enumerate(zip(categories, [lcoh2_mwh, lcoe_mwh, lcoh_mwh])):
    ax.text(i, bottoms[i] + 0.3, f"{lc:.1f} €/MWh",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([r"H$_2$", "Electricity", "Heat"])
ax.set_ylabel("Annualised Cost / [M€/yr]")
ax.set_title("Cost Attribution by Energy Carrier\n(with levelised cost annotations)")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
func.savefigure(fig, "MODEL_cost_attribution_waterfall")
plt.show()


# =============================================================================
# PLOT 8 — Instantaneous Levelised Costs (cost / demand at each timestep)
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

dt = 1  # h per timestep

# ── H2: fixed cost spread uniformly + salt cavern variable cost ───────────────
# Fixed costs attributed to H2: shared supply share + electrolyser H2 frac + salt cavern fixed
cost_h2_fixed_ph  = (cost_supply_h2 + cost_electrolyser_h2 + cost_saltcavern) / len(h2_demand_ts)
saltcavern_var_ts = saltcavern_charge * dt * par.SALTCAVERN_OPEX_COMPRESSOR     # €/h
cost_h2_ts        = pd.Series(cost_h2_fixed_ph, index=h2_demand_ts.index) + saltcavern_var_ts

# ── Electricity: fixed cost spread uniformly + battery + CHP-el variable ──────
cost_el_fixed_ph  = (cost_supply_el + cost_battery + cost_chp_el) / len(el_demand_ts)
cost_el_var_ts    = (bat_charge + bat_discharge) * dt * par.BATTERY_OPEX_VAR / 2  # €/h
cost_el_ts        = pd.Series(cost_el_fixed_ph, index=el_demand_ts.index) + cost_el_var_ts

# ── Heat: fixed cost spread uniformly + heat pump + CHP-heat variable ─────────
cost_heat_fixed_ph = (cost_supply_heat + cost_heatpump
                    + cost_thermalstorage + cost_electrolyser_heat
                    + cost_chp_heat) / len(heat_demand_ts)


cost_heat_var_ts   = heatpump_input * dt * par.HEATPUMP_OPEX_VAR                # €/h
cost_heat_ts       = pd.Series(cost_heat_fixed_ph, index=heat_demand_ts.index) + cost_heat_var_ts

# ── System LCOS: total hourly cost / total hourly demand ──────────────────────
total_demand_ts    = h2_demand_ts + el_demand_ts + heat_demand_ts               # MWh/h
cost_system_ts     = cost_h2_ts + cost_el_ts + cost_heat_ts                     # €/h
total_demand_safe  = total_demand_ts.replace(0, np.nan)

# ── Instantaneous LC timeseries ───────────────────────────────────────────────
h2_demand_safe      = h2_demand_ts.replace(0, np.nan)
el_demand_safe      = el_demand_ts.replace(0, np.nan)
heat_demand_safe    = heat_demand_ts.replace(0, np.nan)

lcoh2_instantaneous = cost_h2_ts   / h2_demand_safe                            # €/MWh_H2
lcoe_instantaneous  = cost_el_ts   / el_demand_safe                            # €/MWh_el
lcoh_instantaneous  = cost_heat_ts / heat_demand_safe                          # €/MWh_th
lcos_instantaneous  = cost_system_ts / total_demand_safe                       # €/MWh

# ── Sanity checks: cost_ts.sum() / demand_ts.sum() must equal scalar LC ───────
# Do NOT use np.mean() — that is a simple average and will NOT match the scalar
# LC unless demand happens to be perfectly flat.
lcoh2_check = cost_h2_ts.sum()     / h2_demand_ts.sum()
lcoe_check  = cost_el_ts.sum()     / el_demand_ts.sum()
lcoh_check  = cost_heat_ts.sum()   / heat_demand_ts.sum()
lcos_check  = cost_system_ts.sum() / total_demand_ts.sum()

print("\n" + "=" * 68)
print("  INSTANTANEOUS LC — SANITY CHECKS (cost_ts.sum()/demand_ts.sum())")
print("  NOTE: np.mean(lc_instantaneous) ≠ scalar LC — use weighted sum.")
print("=" * 68)
print(f"  LCOH2  scalar: {lcoh2_mwh:>8.3f}  reconstructed: {lcoh2_check:>8.3f}  Δ={abs(lcoh2_mwh-lcoh2_check):.5f}")
print(f"  LCOE   scalar: {lcoe_mwh:>8.3f}  reconstructed: {lcoe_check:>8.3f}  Δ={abs(lcoe_mwh-lcoe_check):.5f}")
print(f"  LCOH   scalar: {lcoh_mwh:>8.3f}  reconstructed: {lcoh_check:>8.3f}  Δ={abs(lcoh_mwh-lcoh_check):.5f}")
print(f"  LCOS   scalar: {lcos_mwh:>8.3f}  reconstructed: {lcos_check:>8.3f}  Δ={abs(lcos_mwh-lcos_check):.5f}")
print("=" * 68)

print("Avg LCOH2 (from instantaneous): ", np.mean(lcoh2_instantaneous), "€/MWh")
print("Avg LCOE (from instantaneous): ", np.mean(lcoe_instantaneous), "€/MWh")
print("Avg LCOH (from instantaneous): ", np.mean(lcoh_instantaneous), "€/MWh")
print("Avg LCOS (from instantaneous): ", np.mean(lcos_instantaneous), "€/MWh")
''' #SANITY CHECK: these sums must match the model objective (total cost)
print("Total cost from timeseries: ", cost_system_ts.sum(), "€")
print("Model objective: ", model.objective(), "€")
RESULT:  THEY DO ✅'''


# ── Plot 8a — Instantaneous LCOH2 ────────────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcoh2_instantaneous,
    demand_ts    = h2_demand_ts,
    annual_lc    = lcoh2_mwh,
    lc_label     = r"LCOH$_2$ [€/MWh]",
    demand_label = r"H$_2$ Demand",
    lc_color     = "steelblue",
    demand_color = "steelblue",
    fig_name     = "MODEL_lcoh2_instantaneous",
)

# ── Plot 8b — Instantaneous LCOE ─────────────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcoe_instantaneous,
    demand_ts    = el_demand_ts,
    annual_lc    = lcoe_mwh,
    lc_label     = r"LCOEl [€/MWh]",
    demand_label = r"Electricity Demand",
    lc_color     = "darkorange",
    demand_color = "darkorange",
    fig_name     = "MODEL_lcoe_instantaneous",
)

# ── Plot 8c — Instantaneous LCOH (heat) ──────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcoh_instantaneous,
    demand_ts    = heat_demand_ts,
    annual_lc    = lcoh_mwh,
    lc_label     = r"LCOH [€/MWh]",
    demand_label = r"Heat Demand",
    lc_color     = "firebrick",
    demand_color = "firebrick",
    fig_name     = "MODEL_lcoh_instantaneous",
)

# ── Plot 8d — Instantaneous LCOS (system) ────────────────────────────────────
func.plot_lc_and_demand(
    lc_ts        = lcos_instantaneous,
    demand_ts    = total_demand_ts,
    annual_lc    = lcos_mwh,
    lc_label     = r"LCOS [€/MWh]",
    demand_label = r"Total Demand (all carriers)",
    lc_color     = "purple",
    demand_color = "grey",
    fig_name     = "MODEL_lcos_instantaneous",
)

# ── Plot 8e — LC duration curves (all 4 carriers side by side) ───────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

for ax, lc_ts, annual_lc, label, color in zip(
    axes,
    [lcoh2_instantaneous, lcoe_instantaneous, lcoh_instantaneous, lcos_instantaneous],
    [lcoh2_mwh,           lcoe_mwh,           lcoh_mwh,           lcos_mwh],
    [r"LCOH$_2$",         "LCOE",             "LCOH",             "LCOS"],
    ["steelblue",         "darkorange",        "firebrick",        "purple"],
):
    sorted_vals = lc_ts.dropna().sort_values(ascending=False).values
    ax.plot(sorted_vals, color=color, lw=1.2)
    ax.axhline(annual_lc, color="black", lw=1, ls="--",
               label=f"Annual avg\n{annual_lc:.1f} €/MWh")
    ax.set_title(f"{label} Duration Curve")
    ax.set_xlabel("Hours (ranked)")
    ax.set_ylabel(f"{label} / [€/MWh]")
    ax.legend(fontsize=8)

plt.suptitle("Levelised Cost Duration Curves", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "MODEL_lc_duration_curves")
plt.show()

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
fc_system_bottom              = func.cost_permanent(par.FUELCELL_CAPEX_SYSTEM, invest_sizes["chp"], PROJECT_N, par.WACC)
fc_stack_first, fc_stack_repl = func.split_cost(par.FUELCELL_CAPEX_STACK, invest_sizes["chp"], PROJECT_N, par.WACC, u=par.FUELCELL_LIFETIME_STACK)

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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.4)

# ── Left: stacked bar ─────────────────────────────────────────────────────────
cb = [v / 1e6 for v in cost_bottom]
ct = [v / 1e6 for v in cost_top]

axes[0].bar(x, cb, width, label="Up to 1st replacement",  color="steelblue", hatch="/")
axes[0].bar(x, ct, width, label="Replacements",           color="firebrick", bottom=cb, hatch="-")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Lifetime Costs / [M€]")
#axes[0].set_title(f"Annualised cost breakdown\n(initial vs replacement costs, {PROJECT_N}-yr project)")
axes[0].legend(fontsize=9)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

# ── Right: replacements as % of initial ──────────────────────────────────────
pct = [t / b * 100 if b > 0 else 0.0 for t, b in zip(cost_top, cost_bottom)]
axes[1].bar(x, pct, width, color="firebrick")
for i, p in enumerate(pct):
    axes[1].text(i, p + 0.3, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Replacement / Initial cost [%]")
#axes[1].set_title("Replacement annualised cost\nas % of initial annualised cost")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

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

func.savefigure(fig, "MODEL_replacement_cost_overhead")
plt.show()


# =============================================================================
# FINANCIAL METRICS — NPV, IRR, Payback, H2 Selling Price
# =============================================================================

# ── selling prices ──────────────────────────────────────────────────────────
MARGIN           = par.MARGIN
H2_SELLING_PRICE = lcoh2_mwh / (1 - MARGIN)               # €/MWh_H2
ELECTRICITY_SELLING_PRICE = lcoe_mwh / (1 - MARGIN)        # €/MWh_el
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
    + invest_sizes["chp"]                   * par.FUELCELL_OPEX_FIX
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
    + invest_sizes["chp"]               * par.FUELCELL_CAPEX    
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
print(f"  LCOE               : {lcoe_mwh:>12.2f} €/MWh_el")
print(f"  LCOH               : {lcoh_mwh:>12.2f} €/MWh_th")
print(f"  LCOS (system)      : {lcos_mwh:>12.2f} €/MWh  (all carriers)")
print(f"  Margin             : {MARGIN*100:>10.1f} %")
print(f"  H2 selling price   : {H2_SELLING_PRICE:>10.2f} €/MWh_H2")
print(f"  El selling price   : {ELECTRICITY_SELLING_PRICE:>10.2f} €/MWh_el")
print(f"  Heat selling price : {HEAT_SELLING_PRICE:>10.2f} €/MWh_th")
print(f"  H2 Profit per MWh  : {H2_SELLING_PRICE - lcoh2_mwh:>10.2f} €/MWh_H2")
print(f"  El Profit per MWh  : {ELECTRICITY_SELLING_PRICE - lcoe_mwh:>10.2f} €/MWh_el")
print(f"  Heat Profit per MWh: {HEAT_SELLING_PRICE - lcoh_mwh:>10.2f} €/MWh_th")
print(f"  Annual revenue     : {annual_revenue:>12.0f} €/yr")
print(f"  Annual OPEX total  : {annual_cost_total:>12.0f} €/yr")
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

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(years, npv_series / 1e6, color="steelblue", linewidth=1.0, label="NPV")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

if payback_discounted is not None:
    ax.axvline(payback_discounted, color="black", linewidth=0.8, linestyle="--",
               label=f"Payback = {payback_discounted:.1f} yrs")
    ax.plot(payback_discounted, 0, "o", color="black", markersize=5)

ax.set_xlabel("Year")
ax.set_ylabel("NPV / [M€]")
ax.set_title("Net Present Value over Project Lifetime")
ax.legend()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
func.savefigure(fig, "MODEL_npv_evolution")
plt.show()




# =============================================================================
# FLEXIBILITY METRICS CALCULATION
# =============================================================================


# ── 1. Residual load on the electricity bus ───────────────────────────────────
res_load = fm.residual_load(
    demand = el_demand_ts + el_input + heatpump_input, #FIXME: should this include electrolyser and heat pump demand, and battery charge?
    wind   = wind_flow,
    pv     = pv_flow,
)

# ── 2. Storage metrics ────────────────────────────────────────────────────────
flex_battery = {
    "utilisation_rate":    fm.storage_utilisation_rate(battery_soc, capacity_battery),
    "throughput_cycles":   fm.throughput_ratio(bat_charge, capacity_battery),
    "load_shift_index":    fm.load_shift_index(bat_discharge, capacity_battery),
    **fm.storage_flexibility_band(battery_soc, capacity_battery),
}

flex_saltcavern = {
    "utilisation_rate":    fm.storage_utilisation_rate(saltcavern_soc, capacity_saltcavern),
    "throughput_cycles":   fm.throughput_ratio(saltcavern_charge, capacity_saltcavern),
    "load_shift_index":    fm.load_shift_index(saltcavern_discharge, capacity_saltcavern),
    **fm.storage_flexibility_band(saltcavern_soc, capacity_saltcavern),
}

flex_thermalstorage = {
    "utilisation_rate":    fm.storage_utilisation_rate(thermalstorage_soc, capacity_thermalstorage),
    "throughput_cycles":   fm.throughput_ratio(thermalstorage_charge, capacity_thermalstorage),
    "load_shift_index":    fm.load_shift_index(thermalstorage_discharge, capacity_thermalstorage),
    **fm.storage_flexibility_band(thermalstorage_soc, capacity_thermalstorage),
}

# ── 3. Sector-coupling metrics ────────────────────────────────────────────────
total_el_produced = pv_flow.sum() + wind_flow.sum() + chp_output_el.sum()

flex_electrolyser = fm.electrolyser_flexibility_metrics(
    el_input        = el_input,
    el_output_h2    = el_output,
    el_output_heat  = el_output_heat,
    capacity        = capacity_electrolyser,
    total_el_mwh    = total_el_produced,
    total_h2_mwh    = h2_demand_ts.sum(),
    efficiency_h2   = par.ELECTROLYSER_EFFICIENCY,
    efficiency_heat = par.ELECTROLYSER_RECOVERABLE_HEAT,
)

flex_chp = fm.chp_flexibility_metrics(
    h2_input    = chp_input,
    el_output   = chp_output_el,
    heat_output = chp_output_heat,
    capacity    = capacity_chp,
    total_h2_mwh = h2_demand_ts.sum(),
)

flex_heatpump = {
    "capacity_factor": float(heatpump_input.mean() / capacity_heatpump) if capacity_heatpump > 0 else 0.0,
    "sc_ratio_el_to_heat": fm.sector_coupling_ratio(heatpump_input.sum(), total_el_produced),
    "ramp_up_avg_mw_per_h": float(heatpump_input.diff().clip(lower=0).mean()),
    "ramp_down_avg_mw_per_h": float(heatpump_input.diff().clip(upper=0).abs().mean()),
    "hours_operating": int((heatpump_input > 0).sum()),
}

# ── 4. Demand-side / curtailment metrics ──────────────────────────────────────
flex_curtailment = {
    **fm.curtailment_metrics(curtailment_el,   total_el_produced,   "electricity"),
    **fm.curtailment_metrics(curtailment_h2,   h2_demand_ts.sum(),  "h2"),#FIXME: wrong denominator
    **fm.curtailment_metrics(curtailment_heat, heat_demand_ts.sum(),"heat"), #FIXME: wrong denominator
}

# ── 5. Ramping metrics ────────────────────────────────────────────────────────
flex_ramping = fm.ramping_metrics(res_load)

# ── 6. System-level ratios ────────────────────────────────────────────────────
total_local_gen_el = pv_flow + wind_flow + chp_output_el
flex_system = {
    "ssr_electricity":  fm.self_sufficiency_ratio(el_demand_ts, total_local_gen_el),
    "scr_electricity":  fm.self_consumption_ratio(el_demand_ts, pv_flow + wind_flow),
    "sc_ratio_el_to_h2_lossless":    fm.sector_coupling_ratio(el_input.sum(), total_el_produced),
    "sc_ratio_el_to_h2_with_losses": fm.sector_coupling_ratio(el_input.sum(), total_el_produced,
                                        efficiency=par.ELECTROLYSER_EFFICIENCY, with_losses=True),
    "sc_ratio_el_to_heat_lossless":  fm.sector_coupling_ratio(heatpump_input.sum(), total_el_produced),
    "sc_ratio_h2_to_el_and_heat":    fm.sector_coupling_ratio(chp_input.sum(), h2_demand_ts.sum()), #FIXME:simply wrong
    "res_penetration":  float(total_el_produced / (el_demand_ts.sum() + el_input.sum() + heatpump_input.sum())),
}

# ── 7. Combine into a master dict and print ───────────────────────────────────
all_flex_metrics = {
    "battery":        flex_battery,
    "salt_cavern":    flex_saltcavern,
    "thermal_storage":flex_thermalstorage,
    "electrolyser":   flex_electrolyser,
    "chp":            flex_chp,
    "heat_pump":      flex_heatpump,
    "curtailment":    flex_curtailment,
    "ramping":        flex_ramping,
    "system":         flex_system,
}

flex_table = fm.flexibility_summary_table(all_flex_metrics)

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

# Save to CSV alongside the logbook
flex_table.to_csv(DATA_DIR.parent / f"{SCENARIO_NAME}_flexibility_metrics.csv", index=False)


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
    ax.set_ylabel("SoC / capacity [–]")
    ax.set_ylim(0, 1.05)
    ur = flex_dict["utilisation_rate"]
    li = flex_dict["load_shift_index"]
    tc = flex_dict["throughput_cycles"]
    ax.set_title(
        f"{label}   |   Utilisation rate: {ur:.2%}   "
        f"Total Energy Disharged: {li:.1f} cycles/year   Total Energy Charged: {tc:.1f} cycles/yr",
        fontsize=9,
    )

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].set_xlabel("Month")
plt.suptitle("Normalised State of Charge — All Storage Technologies", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F1_soc_normalised_all_storages")
plt.show()


# =============================================================================
# PLOT F2 — Residual load and ramping pressure
# Purpose : Quantifies how much ramping flexibility the system needs.
#           The wider the ramp distribution, the more flexible assets must be.
#           Over-generation hours indicate curtailment risk.
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── Left: full-year residual load ─────────────────────────────────────────────
axes[0].fill_between(
    res_load.index,
    np.where(res_load > 0, res_load.values, 0),
    alpha=0.6, color="steelblue", label="Deficit (need dispatchable/storage)",
)
axes[0].fill_between(
    res_load.index,
    np.where(res_load < 0, res_load.values, 0),
    alpha=0.6, color="firebrick", label="Surplus (curtailment risk)",
)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_ylabel("Residual Load [MW]")
axes[0].set_xlabel("Month")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[0].legend(fontsize=8)
axes[0].set_title(
    f"Residual Load (demand – wind – PV)\n"
    f"Over-gen hours: {flex_ramping['over_gen_hours']}  "
    f"({flex_ramping['over_gen_fraction']:.1%} of year)"
)

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
axes[1].legend(fontsize=8)

# ── Right: residual load duration curve ──────────────────────────────────────
rl_sorted = res_load.sort_values(ascending=False).values
axes[2].plot(rl_sorted, color="steelblue", lw=1.2)
axes[2].fill_between(
    range(len(rl_sorted)),
    np.where(rl_sorted > 0, rl_sorted, 0),
    alpha=0.4, color="steelblue", label="Must cover",
)
axes[2].fill_between(
    range(len(rl_sorted)),
    np.where(rl_sorted < 0, rl_sorted, 0),
    alpha=0.4, color="firebrick", label="Surplus",
)
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set_xlabel("Hours (ranked)")
axes[2].set_ylabel("Residual Load [MW]")
axes[2].set_title(
    f"Residual Load Duration Curve\n"
    f"Range: {flex_ramping['residual_load_range_mw']:.1f} MW"
)
axes[2].legend(fontsize=8)

plt.suptitle("Load Characteristics on the Electricity Bus", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F2_residual_load_ramping")
plt.show()


# =============================================================================
# PLOT F3 — Sector-coupling dispatch: where does electricity go each hour?
# Purpose : Shows how the electrolyser and heat pump compete for electricity,
#           and how the CHP feeds electricity back. The stacked area reveals
#           which sector-coupling pathway dominates at each time of year.
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# ── Top: electricity generation and use ───────────────────────────────────────
gen_total = wind_flow + pv_flow + chp_output_el
base = np.zeros(len(gen_total))

axes[0].fill_between(gen_total.index, base, wind_flow.values,
                     label="Wind", color="tab:blue", alpha=0.75)
base += wind_flow.values
axes[0].fill_between(gen_total.index, base, base + pv_flow.values,
                     label="PV", color="tab:orange", alpha=0.75)
base += pv_flow.values
axes[0].fill_between(gen_total.index, base, base + chp_output_el.values,
                     label="CHP → Electricity", color="tab:green", alpha=0.75)

axes[0].plot(el_demand_ts.index,   el_demand_ts.values,   color="black",  lw=1.2, ls="--", label="El demand")
axes[0].plot(el_input.index,       el_input.values,       color="purple", lw=1.0,           label="Electrolyser")
axes[0].plot(heatpump_input.index, heatpump_input.values, color="red",    lw=1.0,           label="Heat Pump")
axes[0].set_ylabel("MW")
axes[0].legend(loc="upper right", fontsize=7, ncol=3)
axes[0].set_title("Electricity Bus: Generation and Sector-Coupling Dispatch")

# ── Middle: H2 bus ────────────────────────────────────────────────────────────
base = np.zeros(len(el_output))
axes[1].fill_between(el_output.index, base, el_output.values,
                     label="Electrolyser → H₂", color="tab:blue", alpha=0.75)
axes[1].plot(chp_input.index,    chp_input.values,    color="black",  lw=1.2, ls="--", label="CHP demand (H₂)")
axes[1].plot(h2_demand_ts.index, h2_demand_ts.values, color="orange", lw=1.2, ls=":",  label="H₂ end demand")
axes[1].set_ylabel("MW")
axes[1].legend(loc="upper right", fontsize=7, ncol=3)
axes[1].set_title("H₂ Bus: Production and Consumption")

# ── Bottom: heat bus ──────────────────────────────────────────────────────────
base = np.zeros(len(heatpump_output))
axes[2].fill_between(heatpump_output.index, base, heatpump_output.values,
                     label="Heat Pump", color="tab:red", alpha=0.75)
base += heatpump_output.values
axes[2].fill_between(el_output_heat.index, base, base + el_output_heat.values,
                     label="Electrolyser waste heat", color="tab:purple", alpha=0.75)
base += el_output_heat.values
axes[2].fill_between(chp_output_heat.index, base, base + chp_output_heat.values,
                     label="CHP heat", color="tab:green", alpha=0.75)
axes[2].plot(heat_demand_ts.index, heat_demand_ts.values, color="black", lw=1.2, ls="--", label="Heat demand")
axes[2].set_ylabel("MW")
axes[2].legend(loc="upper right", fontsize=7, ncol=2)
axes[2].set_title("Heat Bus: Production and Demand")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[2].set_xlabel("Month")

plt.suptitle("Sector-Coupling Dispatch Across All Three Energy Buses", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F3_sector_coupling_dispatch_full_year")
plt.show()


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
#   - CHP (converting H2 back to electricity when needed)
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
    "CHP (H2 → El+Heat)":        chp_input.values,
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
    "CHP (H2 → El+Heat)",
]
failure_cols = ["Curtailment El", "Curtailment H2", "Curtailment Heat"]

supply_colors = [
    "purple", "brown", "orange", "tab:blue", "tab:red", "tab:green",
]
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
axes[0].set_title("Monthly Flexibility Provision by Source")
axes[0].legend(loc="upper right", fontsize=8, ncol=2)

# ── Bottom: curtailment (flexibility failure) ─────────────────────────────────
bottoms = np.zeros(len(monthly_flex))
for col, color in zip(failure_cols, failure_colors):
    vals = monthly_flex[col].values / 1e3  # GWh
    axes[1].bar(x, vals, bottom=bottoms / 1e3, label=col, color=color, width=0.8, alpha=0.8)
    bottoms += monthly_flex[col].values

axes[1].set_ylabel("Curtailment [GWh/month]")
axes[1].set_title("Monthly Flexibility Failure (Curtailment by Carrier)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [d.strftime("%b") for d in monthly_flex.index], rotation=45
)
axes[1].legend(loc="upper right", fontsize=8)

plt.suptitle("Monthly Flexibility: Provision vs Failure", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F4_monthly_flexibility_provision_vs_failure")
plt.show()



# =============================================================================
# PLOT F5 — Electrolyser load duration curve with flexibility zones
# Purpose : The shape of the electrolyser load duration curve reveals how
#           flexibly it operates. A flat curve = baseload (inflexible).
#           A steep curve = highly variable (flexible, absorbing surplus).
#           Same logic applies to CHP and heat pump.
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, flow, cap, metrics_d, label, color in zip(
    axes,
    [el_input,         chp_input,         heatpump_input],
    [capacity_electrolyser, capacity_chp,  capacity_heatpump],
    [flex_electrolyser, flex_chp,          flex_heatpump],
    ["Electrolyser",   "CHP",             "Heat Pump"],
    ["tab:blue",       "tab:green",       "tab:red"],
):
    if cap <= 0:
        ax.set_title(f"{label}\n(not installed)")
        continue

    sorted_load   = flow.sort_values(ascending=False).values
    norm_load     = sorted_load / cap
    cf            = metrics_d.get("capacity_factor", 0.0)
    hours_op      = int((flow > 0).sum())

    #ax.fill_between(range(len(norm_load)), norm_load, alpha=0.5, color=color)
    ax.plot(range(len(norm_load)), norm_load, color=color, lw=1.2)
    ax.axhline(cf, color="black", lw=1.2, ls="--",
               label=f"Capacity factor: {cf:.2%}")
    ax.set_xlabel("Hours (ranked)")
    ax.set_ylabel("Load / capacity [–]")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{label}\nCF: {cf:.2%}   Operating hours: {hours_op}/8760"
    )
    ax.legend(fontsize=8)

plt.suptitle("Load Duration Curves — Sector-Coupling Technologies\n"
             "(steeper = more flexible / variable operation)", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX_F5_converter_load_duration_curves")
plt.show()


# =============================================================================
# PLOT F6 — Storage cycling: charge vs discharge scatter
# Purpose : Each point is one hour. Clustering near the axes indicates the
#           storage mostly charges OR discharges (not both simultaneously,
#           which would be a model artefact). The spread shows flexibility range.
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, charge, discharge, cap, label, color in zip(
    axes,
    [bat_charge,      saltcavern_charge,      thermalstorage_charge],
    [bat_discharge,   saltcavern_discharge,   thermalstorage_discharge],
    [invest_sizes["battery_power"], invest_sizes["saltcavern_power"],   invest_sizes["thermalstorage_energy"]],
    ["Battery",       "Salt Cavern",          "Thermal Storage"],
    ["purple",        "brown",                "orange"],
):
    if cap <= 0:
        ax.set_title(f"{label}\n(not installed)")
        continue

    ax.scatter(
        charge.values / cap,
        discharge.values / cap,
        alpha=0.15, s=4, color=color,
    )
    ax.set_xlabel("Charge rate / capacity [–]")
    ax.set_ylabel("Discharge rate / capacity [–]")
    ax.set_title(f"{label}\nCharge vs Discharge (hourly)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.5, ls="--", alpha=0.4)

plt.suptitle("Storage Charge vs Discharge Rates (hourly scatter)\n"
             "Points near axes = clean single-direction operation each hour", fontsize=12)
plt.tight_layout()
func.savefigure(fig, "FLEX_F6_storage_charge_discharge_scatter")
plt.show()



# =============================================================================
# PLOT F7 — Sector Coupling: supply fraction and absolute energy
# =============================================================================
# 4 bars: El→H2 | El→Heat | H2→El | H2→Heat
#
# Colour logic — one colour family per technology:
#   Electrolyser  → blue family  (H2 output: steelblue, waste heat: lightsteelblue)
#   Heat Pump     → green family (heat output: seagreen)
#   CHP           → red family   (electricity: firebrick, heat: lightcoral)
#
# Bar 1 "El→H2"   : electrolyser H2 output only          → steelblue
# Bar 2 "El→Heat" : HP output (seagreen) STACKED ON TOP
#                   electrolyser waste heat (lightsteelblue)
# Bar 3 "H2→El"   : CHP electricity output only          → firebrick
# Bar 4 "H2→Heat" : CHP heat output only                 → lightcoral

# ── Colour palette ────────────────────────────────────────────────────────────
C_ELEC_H2    = "steelblue"       # electrolyser → H2
C_ELEC_HEAT  = "steelblue"  # electrolyser → waste heat
C_HP         = "seagreen"        # heat pump → heat
C_CHP_EL     = "firebrick"       # CHP → electricity
C_CHP_HT     = "firebrick"      # CHP → heat

# ── Supply denominators ───────────────────────────────────────────────────────
total_res_gen = pv_flow.sum() + wind_flow.sum()
total_h2_gen  = el_output.sum()

# ── Individual component values ───────────────────────────────────────────────

# Bar 1 — El → H2 (electrolyser H2 output only)
ratio_elec_h2    = el_output.sum()       / total_res_gen * 100 if total_res_gen > 0 else 0.0
energy_elec_h2   = float(el_output.sum() / 1e3)

# Bar 2 — El → Heat (electrolyser waste heat + heat pump, stacked)
ratio_elec_waste = el_output_heat.sum()  / total_res_gen * 100 if total_res_gen > 0 else 0.0
ratio_hp         = heatpump_output.sum() / total_res_gen * 100 if total_res_gen > 0 else 0.0
energy_elec_waste= float(el_output_heat.sum()  / 1e3)
energy_hp        = float(heatpump_output.sum() / 1e3)

# Bar 3 — H2 → El (CHP electricity output only)
ratio_chp_el     = chp_output_el.sum()   / total_h2_gen * 100 if total_h2_gen > 0 else 0.0
energy_chp_el    = float(chp_output_el.sum()   / 1e3)

# Bar 4 — H2 → Heat (CHP heat output only)
ratio_chp_heat   = chp_output_heat.sum() / total_h2_gen * 100 if total_h2_gen > 0 else 0.0
energy_chp_heat  = float(chp_output_heat.sum() / 1e3)

# ── Bar positions ─────────────────────────────────────────────────────────────
x_pos      = np.array([0, 1, 2, 3])
bar_width  = 0.5
bar_labels = [
    "El → H₂\n(Electrolyser)",
    "El → Heat\n(Electrolyser waste\n+ Heat Pump)",
    "H₂ → El\n(CHP)",
    "H₂ → Heat\n(CHP)",
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Helper: annotate bar top and (optionally) inner segment
def annotate_top(ax, x, total, unit=""):
    ax.text(x, total + ax.get_ylim()[1] * 0.01,
            f"{total:.1f}{unit}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

def annotate_inner(ax, x, y_centre, text):
    """Only annotate if segment is tall enough to fit text."""
    if y_centre > ax.get_ylim()[1] * 0.03:
        ax.text(x, y_centre, text,
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")


# LEFT — Sector coupling ratios [%] ─────────────────────────────────────────────────────────────

# Bar 1: single segment
axes[0].bar(x_pos[0], ratio_elec_h2, bar_width,
            color=C_ELEC_H2, alpha=0.88, edgecolor="white",
            label="Electrolyser → H₂")

# Bar 2: stacked — electrolyser waste heat (bottom) + HP (top)
axes[0].bar(x_pos[1], ratio_elec_waste, bar_width,
            color=C_ELEC_HEAT, alpha=0.88, edgecolor="white",
            label="Electrolyser → Waste heat")
axes[0].bar(x_pos[1], ratio_hp, bar_width,
            bottom=ratio_elec_waste,
            color=C_HP, alpha=0.88, edgecolor="white",
            label="Heat Pump → Heat")

# Bar 3: single segment
axes[0].bar(x_pos[2], ratio_chp_el, bar_width,
            color=C_CHP_EL, alpha=0.88, edgecolor="white",
            label="CHP → Electricity")

# Bar 4: single segment
axes[0].bar(x_pos[3], ratio_chp_heat, bar_width,
            color=C_CHP_HT, alpha=0.88, edgecolor="white",
            label="CHP → Heat")

# top annotations
axes[0].set_ylim(0, max(ratio_elec_h2,
                        ratio_elec_waste + ratio_hp,
                        ratio_chp_el,
                        ratio_chp_heat) * 1.22)

annotate_top(axes[0], x_pos[0], ratio_elec_h2,              unit="%")
annotate_top(axes[0], x_pos[1], ratio_elec_waste + ratio_hp, unit="%")
annotate_top(axes[0], x_pos[2], ratio_chp_el,               unit="%")
annotate_top(axes[0], x_pos[3], ratio_chp_heat,             unit="%")

# inner annotations for bar 2
annotate_inner(axes[0], x_pos[1],
               ratio_elec_waste / 2,
               f"Waste\n{ratio_elec_waste:.1f}%")
annotate_inner(axes[0], x_pos[1],
               ratio_elec_waste + ratio_hp / 2,
               f"HP\n{ratio_hp:.1f}%")

axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(bar_labels, fontsize=9)
axes[0].set_ylabel("% of carrier supply crossing sector boundary")
axes[0].set_title(
    "Sector Coupling Ratio\n"
    "Bars 1–2: % of RES electricity   |   Bars 3–4: % of H₂ produced",
    fontsize=9,
)
axes[0].axhline(100, color="grey", lw=0.6, ls="--", alpha=0.4)
axes[0].legend(fontsize=8, loc="upper right")


# RIGHT — Absolute annual energy [GWh] ─────────────────────────────────────────────────────────────


# Bar 1: single segment
axes[1].bar(x_pos[0], energy_elec_h2, bar_width,
            color=C_ELEC_H2, alpha=0.88, edgecolor="white",
            label="Electrolyser → H₂")

# Bar 2: stacked — electrolyser waste heat (bottom) + HP (top)
axes[1].bar(x_pos[1], energy_elec_waste, bar_width,
            color=C_ELEC_HEAT, alpha=0.88, edgecolor="white",
            label="Electrolyser → Waste heat")
axes[1].bar(x_pos[1], energy_hp, bar_width,
            bottom=energy_elec_waste,
            color=C_HP, alpha=0.88, edgecolor="white",
            label="Heat Pump → Heat")

# Bar 3: single segment
axes[1].bar(x_pos[2], energy_chp_el, bar_width,
            color=C_CHP_EL, alpha=0.88, edgecolor="white",
            label="CHP → Electricity")

# Bar 4: single segment
axes[1].bar(x_pos[3], energy_chp_heat, bar_width,
            color=C_CHP_HT, alpha=0.88, edgecolor="white",
            label="CHP → Heat")

# top annotations
axes[1].set_ylim(0, max(energy_elec_h2,
                        energy_elec_waste + energy_hp,
                        energy_chp_el,
                        energy_chp_heat) * 1.22)

annotate_top(axes[1], x_pos[0], energy_elec_h2,               unit=" GWh")
annotate_top(axes[1], x_pos[1], energy_elec_waste + energy_hp, unit=" GWh")
annotate_top(axes[1], x_pos[2], energy_chp_el,                unit=" GWh")
annotate_top(axes[1], x_pos[3], energy_chp_heat,              unit=" GWh")

# inner annotations for bar 2
annotate_inner(axes[1], x_pos[1],
               energy_elec_waste / 2,
               f"Waste\n{energy_elec_waste:.1f} GWh")
annotate_inner(axes[1], x_pos[1],
               energy_elec_waste + energy_hp / 2,
               f"HP\n{energy_hp:.1f} GWh")

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(bar_labels, fontsize=9)
axes[1].set_ylabel("Annual energy output crossing boundary [GWh]")
axes[1].set_title(
    "Absolute Cross-Sector Energy Flows\n"
    "(outputs delivered to destination carrier)",
    fontsize=9,
)
axes[1].legend(fontsize=8, loc="upper right")

plt.suptitle(
    "Sector Coupling — Supply Fraction and Absolute Energy Crossing Sector Boundaries",
    fontsize=12,
)
plt.tight_layout()
func.savefigure(fig, "FLEX_F7_sector_coupling_ratios")
plt.show()

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
#TODO: Refine parameters. Merge "Capacity factor" and "Storage Util."
#TODO: Alternative is split into two radars: one for storage, one for sector coupling.
#TODO: up and downward flex for PtX  can be done via max and min loads relative to current load
#TODO: Verify formulas utilised in "radar_raw".

epsilon = 1e-6

def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def responsiveness(flow: pd.Series, capacity: float, eps: float = epsilon) -> float:
    """Fraction of online hours with active ramping."""
    if capacity <= 0:
        return 0.0
    threshold    = eps * capacity
    online       = flow > threshold
    ramp         = flow.diff().abs() > threshold
    online_hours = online.sum()
    if online_hours == 0:
        return 0.0
    return float((online & ramp).sum() / online_hours)

def ptx_upward_flex(flow: pd.Series, capacity: float) -> float:
    """Average headroom / capacity — how much more the component could absorb."""
    if capacity <= 0:
        return 0.0
    return float((capacity - flow).clip(lower=0).mean() / capacity)

def ptx_downward_flex(flow: pd.Series, capacity: float,
                      min_load_fraction: float = 0.0) -> float:
    """Average footroom / capacity — how much the component could reduce."""
    if capacity <= 0:
        return 0.0
    min_load_abs = min_load_fraction * capacity
    online       = flow > min_load_abs
    if online.sum() == 0:
        return 0.0
    return float((flow - min_load_abs).clip(lower=0)[online].mean() / capacity)

def ramp_utilisation(flow: pd.Series, capacity: float, min_load_fraction: float = 0.0) -> float:
    """
    Mean absolute hourly ramp / capacity.
    Measures how intensely the component modulates, not just how often.
    Natural PtX equivalent of the storage load-shift index.
    Bounded [0, 1]: 1 would mean swinging full capacity every single hour.
    """
    if capacity <= 0:
        return 0.0
    min_load_abs      = min_load_fraction * capacity
    controllable_range = capacity - min_load_abs
    if controllable_range <= 0:
        return 0.0
    online_mask = flow > min_load_abs
    ramps_online = flow.diff().abs()[online_mask]
    if ramps_online.empty:
        return 0.0
    return float(ramps_online.mean() / controllable_range)

# ── Storage metrics ────────────────────────────────────────────────────────────
n_hours = len(bat_charge)

bat_utilisation = (battery_soc.mean()        / capacity_battery)        if capacity_battery        > 0 else 0.0
sc_utilisation  = (saltcavern_soc.mean()     / capacity_saltcavern)     if capacity_saltcavern     > 0 else 0.0
ts_utilisation  = (thermalstorage_soc.mean() / capacity_thermalstorage) if capacity_thermalstorage > 0 else 0.0

bat_lsi = min(bat_discharge/capacity_battery*len(bat_discharge), 1.0)
sc_lsi  = min(saltcavern_discharge/capacity_saltcavern*len(saltcavern_discharge), 1.0)
ts_lsi  = min(thermalstorage_discharge/capacity_thermalstorage*len(thermalstorage_discharge), 1.0)

bat_up  = _safe(flex_battery["upward_mwh"])        / (capacity_battery        + 1e-9)
sc_up   = _safe(flex_saltcavern["upward_mwh"])     / (capacity_saltcavern     + 1e-9)
ts_up   = _safe(flex_thermalstorage["upward_mwh"]) / (capacity_thermalstorage + 1e-9)

bat_dn  = _safe(flex_battery["downward_mwh"])        / (capacity_battery        + 1e-9)
sc_dn   = _safe(flex_saltcavern["downward_mwh"])     / (capacity_saltcavern     + 1e-9)
ts_dn   = _safe(flex_thermalstorage["downward_mwh"]) / (capacity_thermalstorage + 1e-9)

bat_resp = responsiveness(bat_discharge + bat_charge,                   capacity_battery)
sc_resp  = responsiveness(saltcavern_discharge + saltcavern_charge,     capacity_saltcavern)
ts_resp  = responsiveness(thermalstorage_discharge + thermalstorage_charge, capacity_thermalstorage)

# ── PtX metrics ────────────────────────────────────────────────────────────────
el_cf  = (el_input.mean()       / capacity_electrolyser) if capacity_electrolyser > 0 else 0.0
hp_cf  = (heatpump_input.mean() / capacity_heatpump)     if capacity_heatpump     > 0 else 0.0
chp_cf = (chp_input.mean()      / capacity_chp)          if capacity_chp          > 0 else 0.0

el_ramp_util  = ramp_utilisation(el_input,       capacity_electrolyser, par.ELECTROLYSER_MIN_LOAD)
hp_ramp_util  = ramp_utilisation(heatpump_input, capacity_heatpump,     par.HEATPUMP_MIN_LOAD)
chp_ramp_util = ramp_utilisation(chp_input,      capacity_chp,          par.FUELCELL_MIN_LOAD)

el_up  = ptx_upward_flex(el_input,       capacity_electrolyser)
hp_up  = ptx_upward_flex(heatpump_input, capacity_heatpump)
chp_up = ptx_upward_flex(chp_input,      capacity_chp)

el_dn  = ptx_downward_flex(el_input,       capacity_electrolyser, par.ELECTROLYSER_MIN_LOAD)
hp_dn  = ptx_downward_flex(heatpump_input, capacity_heatpump,     par.HEATPUMP_MIN_LOAD)
chp_dn = ptx_downward_flex(chp_input,      capacity_chp,          par.FUELCELL_MIN_LOAD)

el_resp  = responsiveness(el_input,       capacity_electrolyser)
hp_resp  = responsiveness(heatpump_input, capacity_heatpump)
chp_resp = responsiveness(chp_input,      capacity_chp)



# ── Radar data ─────────────────────────────────────────────────────────────────
storage_radar = {
    "Battery": {
        "Utilisation\n(SoC avg)": min(bat_utilisation, 1.0),
        "Load-shift\nindex":      bat_lsi,
        "Upward\nflexibility":    min(bat_up,  1.0),
        "Downward\nflexibility":  min(bat_dn,  1.0),
        "Responsiveness":         bat_resp,
    },
    "Salt Cavern": {
        "Utilisation\n(SoC avg)": min(sc_utilisation,  1.0),
        "Load-shift\nindex":      sc_lsi,
        "Upward\nflexibility":    min(sc_up,   1.0),
        "Downward\nflexibility":  min(sc_dn,   1.0),
        "Responsiveness":         sc_resp,
    },
    "Thermal Storage": {
        "Utilisation\n(SoC avg)": min(ts_utilisation,  1.0),
        "Load-shift\nindex":      ts_lsi,
        "Upward\nflexibility":    min(ts_up,   1.0),
        "Downward\nflexibility":  min(ts_dn,   1.0),
        "Responsiveness":         ts_resp,
    },
}

ptx_radar = {
    "Electrolyser": {
        "Capacity\nfactor":          min(el_cf,        1.0),
        "Avg Δ (x10)":               min(el_ramp_util, 1.0),  # scale: avg 10% ramp/h = 1.0
        "Upward\nflexibility":       min(el_up,        1.0),
        "Downward\nflexibility":     min(el_dn,        1.0),
        "Responsiveness":            el_resp,
    },
    "Heat Pump": {
        "Capacity\nfactor":          min(hp_cf,        1.0),
        "Avg Δ (x10)":               min(hp_ramp_util, 1.0),
        "Upward\nflexibility":       min(hp_up,        1.0),
        "Downward\nflexibility":     min(hp_dn,        1.0),
        "Responsiveness":            hp_resp,
    },
    "CHP": {
        "Capacity\nfactor":          min(chp_cf,       1.0),
        "Avg Δ (x10)":               min(chp_ramp_util, 1.0),
        "Upward\nflexibility":       min(chp_up,       1.0),
        "Downward\nflexibility":     min(chp_dn,       1.0),
        "Responsiveness":            chp_resp,
    },
}

# ── Plot helper ────────────────────────────────────────────────────────────────
def draw_radar(ax, data: dict, colors: list, title: str):
    categories = list(list(data.values())[0].keys())
    N          = len(categories)
    angles     = [n / float(N) * 2 * np.pi for n in range(N)]
    angles    += angles[:1]

    for (tech, vals), color in zip(data.items(), colors):
        values  = [vals[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, lw=1.8, color=color, label=tech)
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), fontsize=9)
    ax.set_title(title, fontsize=10, pad=20)

fig, axes = plt.subplots(
    1, 2, figsize=(16, 7),
    subplot_kw=dict(polar=True),
)

draw_radar(
    axes[0], storage_radar,
    colors=["purple", "brown", "orange"],
    title=(
        "Storage Flexibility\n"
        "Utilisation = mean SoC / capacity\n"
        "Up/Down flex = avg ramp rate / capacity"
    ),
)

draw_radar(
    axes[1], ptx_radar,
    colors=["tab:blue", "tab:green", "tab:red"],
    title=(
        "PtX Converter Flexibility\n"
        "Ramp util = mean |ΔP| / capacity  (10% ramp/h → 1.0)\n"
    ),
)

fig.suptitle(
    "Flexibility Radar Charts — Storage (left) vs PtX Converters (right)\n"
    "All axes normalised to [0, 1]",
    fontsize=12, y=1.02,
)

plt.tight_layout()
func.savefigure(fig, "FLEX_F8_flexibility_radar_chart")
plt.show()

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
# PLOT F9 — Weekly rolling flexibility intensity
# Purpose : Shows how the balance between storage-based and sector-coupling
#           flexibility shifts over the year. A 7-day rolling mean smooths
#           out hourly noise while preserving seasonal trends.
# =============================================================================

weekly_storage_flex = (
    bat_discharge + saltcavern_discharge + thermalstorage_discharge
).rolling(24 * 7, min_periods=1).mean()

weekly_sc_flex = (
    el_input + heatpump_input + chp_input
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
plt.show()


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

# Save as CSV (already done above via flex_table)
print(f"\n  Flexibility metrics saved to: "
      f"[03] Results/{SCENARIO_NAME}_flexibility_metrics.csv")



# =============================================================================
# LOGBOOK — CSV
# =============================================================================

logbook = {

    "scenario": SCENARIO_NAME,

    # ── Key model parameters ──────────────────────────────────────────────────
    "parameters": {
        "wacc":                              par.WACC,

        "general_lifetime":                  par.LIFETIME,

        "margin":                         par.MARGIN,

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

        "chp_efficiency_el":                 par.FUELCELL_EFFICIENCY_ELECTRICITY,
        "chp_efficiency_heat":               par.FUELCELL_EFFICIENCY_HEAT,
        "chp_min_load":                      par.FUELCELL_MIN_LOAD,
        "chp_lifetime_system":               par.FUELCELL_LIFETIME_SYSTEM,
        "chp_lifetime_stack":                par.FUELCELL_LIFETIME_STACK,

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

        "capacity_chp_mw":                   float(capacity_chp),

        "capacity_heatpump_mw":              float(capacity_heatpump),

        "capacity_battery_energy_mwh":       float(capacity_battery),
        "capacity_battery_power_mw":         float(invest_sizes["battery_power"]),

        "capacity_saltcavern_energy_mwh":    float(capacity_saltcavern),
        "capacity_saltcavern_compressor_mw": float(invest_sizes["saltcavern_power"]),

        "capacity_thermalstorage_mwh":       float(capacity_thermalstorage),
    },

    # ── Operational metrics ───────────────────────────────────────────────────
    "operational": {
        "capacity_factor_pv":                func.capacity_factor(pv_flow,   capacity_pv,   dt) if capacity_pv   > 0 else 0.0,
        "capacity_factor_wind":              func.capacity_factor(wind_flow, capacity_wind, dt) if capacity_wind > 0 else 0.0,
        "capacity_factor_res_avg":           func.capacity_factor(pv_flow + wind_flow, capacity_pv + capacity_wind, dt) if (capacity_pv + capacity_wind) > 0 else 0.0,
        "capacity_factor_electrolyser":      func.capacity_factor(el_input, capacity_electrolyser, dt) if capacity_electrolyser > 0 else 0.0,
        "electrolyser_frac_h2":              float(electro_frac_h2),
        "electrolyser_frac_heat":            float(electro_frac_heat),
        "capacity_factor_chp":               func.capacity_factor(chp_input, capacity_chp, dt) if capacity_chp > 0 else 0.0,
        "chp_frac_el":                       float(chp_frac_el),
        "chp_frac_heat":                     float(chp_frac_heat),
        "capacity_factor_heatpump":          func.capacity_factor(heatpump_input, capacity_heatpump, dt) if capacity_heatpump > 0 else 0.0,
        "total_h2_produced_mwh":             float(total_h2_mwh),
        "total_h2_produced_kg":              float(total_h2_kg),
        "total_heat_produced_mwh":           float(total_heat_mwh),
        "total_el_served_mwh":               float(el_demand_ts.sum()),
        "total_renewable_gen_mwh":           float(pv_flow.sum() + wind_flow.sum()),
        "curtailment_el_mwh":                float(curtailment_el.sum()),
        "curtailment_h2_mwh":                float(curtailment_h2.sum()),
        "curtailment_heat_mwh":              float(curtailment_heat.sum()),
        "curtailment_el_fraction":           float(curtailment_el.sum() / (pv_flow.sum() + wind_flow.sum())) if (pv_flow.sum() + wind_flow.sum()) > 0 else 0.0,
        "curtailment_h2_fraction":           float(curtailment_h2.sum() / total_h2_mwh) if total_h2_mwh > 0 else 0.0,
        "curtailment_heat_fraction":         float(curtailment_heat.sum() / total_heat_mwh) if total_heat_mwh > 0 else 0.0,
        "capacity_factor_battery_energy":    func.capacity_factor(battery_soc,   invest_sizes["battery_energy"], dt) if invest_sizes["battery_energy"] > 0 else 0.0,
        "capacity_factor_battery_power":     func.capacity_factor(bat_discharge, invest_sizes["battery_power"],  dt) if invest_sizes["battery_power"]  > 0 else 0.0,
        "battery_charge_mwh":                float(bat_charge.sum()),
        "battery_discharge_mwh":             float(bat_discharge.sum()),
        "capacity_factor_saltcavern_energy": func.capacity_factor(saltcavern_soc,       invest_sizes["saltcavern_energy"], dt) if invest_sizes["saltcavern_energy"] > 0 else 0.0,
        "capacity_factor_saltcavern_power":  func.capacity_factor(saltcavern_discharge, invest_sizes["saltcavern_power"],  dt) if invest_sizes["saltcavern_power"]  > 0 else 0.0,
        "saltcavern_charge_mwh":             float(saltcavern_charge.sum()),
        "saltcavern_discharge_mwh":          float(saltcavern_discharge.sum()),
        "capacity_factor_thermalstorage_energy": func.capacity_factor(thermalstorage_soc,       invest_sizes["thermalstorage_energy"], dt) if invest_sizes["thermalstorage_energy"] > 0 else 0.0,
        "capacity_factor_thermalstorage_power":  func.capacity_factor(thermalstorage_discharge, invest_sizes["thermalstorage_energy"], dt) if invest_sizes["thermalstorage_energy"] > 0 else 0.0,
        "thermalstorage_charge_mwh":         float(thermalstorage_charge.sum()),
        "thermalstorage_discharge_mwh":      float(thermalstorage_discharge.sum()),
    },

    # ── Levelised costs ───────────────────────────────────────────────────────
    "levelized_cost": {
        "lcoh2_eur_per_mwh":   float(lcoh2_mwh),
        "lcoh2_eur_per_kg":    float(lcoh2_kg),
        "lcoe_eur_per_mwh":    float(lcoe_mwh),
        "lcoh_eur_per_mwh":    float(lcoh_mwh),
        "lcos_eur_per_mwh":    float(lcos_mwh),
    },

    # ── Commercial ────────────────────────────────────────────────────────────
    "commercial": {
        "margin_applied":                        float(MARGIN),
        "h2_selling_price_eur_per_mwh":          float(H2_SELLING_PRICE),
        "h2_profit_eur_per_mwh":                 float(H2_SELLING_PRICE - lcoh2_mwh),
        "electricity_selling_price_eur_per_mwh": float(ELECTRICITY_SELLING_PRICE),
        "electricity_profit_eur_per_mwh":        float(ELECTRICITY_SELLING_PRICE - lcoe_mwh),
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

        "chp_system_eur":            float(invest_sizes["chp"]               * par.FUELCELL_CAPEX_SYSTEM),
        "chp_stack_eur":             float(invest_sizes["chp"]               * par.FUELCELL_CAPEX_STACK),
        "chp_total_eur":             float(invest_sizes["chp"]               * par.FUELCELL_CAPEX),

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
        "chp":            float(invest_sizes["chp"]               * par.FUELCELL_OPEX_FIX),
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



func.save_logbook_csv(logbook, DATA_DIR, SCENARIO_NAME)

