'''Author: Gonçalo Costa Pina
Date_Created: 2026-01-30 (30th January 2026)
Date_Modified: 2026-04-23

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
# FLEXIBILITY METRICS & VISUALISATIONS
# =============================================================================

# ── 0. Residual loads on each bus ─────────────────────────────────────────────
# Electricity bus: net demand after subtracting non-dispatchable RES
# Positive = deficit (needs ramping up); Negative = surplus (store or curtail)
residual_el = (el_demand_ts
               + el_input          # electrolyser consumes from el bus
               + heatpump_input    # heatpump consumes from el bus
               - (pv_flow + wind_flow)# - curtailment_el)
               - chp_output_el
               ).values

# H2 bus: net demand after subtracting electrolyser output
residual_h2 = (h2_demand_ts #+ curtailment_h2
               - el_output
               + chp_input         # CHP consumes H2
               ).values

# Heat bus: net demand after subtracting all heat producers
residual_heat = (heat_demand_ts #+ curtailment_heat
                 - el_output_heat   # electrolyser waste heat
                 - chp_output_heat  # CHP heat output
                 - heatpump_output  # heat pump output
                 ).values

residual_el_ts   = pd.Series(residual_el,   index=idx)
residual_h2_ts   = pd.Series(residual_h2,   index=idx)
residual_heat_ts = pd.Series(residual_heat, index=idx)

# ── 1. Ramping metrics (1h, 3h, 8h differences) ───────────────────────────────
# These follow the ENTSO-E / PATHFNDR methodology:
# Large positive ramps = sudden need for upward flexibility
# Large negative ramps = sudden need for downward flexibility

def compute_ramp_stats(ts, label, windows=[1, 3, 8]):
    """Compute ramp statistics for multiple time windows."""
    stats = {}
    for w in windows:
        #For the w-hour ramp, we compute the difference between time t and time t-w
        #Key results (statistics) are printed in the console and can be used for comparison across carriers or scenarios.
        ramp = ts.diff(w).dropna()
        stats[f"{label}_ramp_{w}h_max_up"]   = ramp.max()
        stats[f"{label}_ramp_{w}h_max_down"]  = ramp.min()
        stats[f"{label}_ramp_{w}h_std"]       = ramp.std()
        stats[f"{label}_ramp_{w}h_p95_up"]    = ramp.quantile(0.95)
        stats[f"{label}_ramp_{w}h_p95_down"]  = ramp.quantile(0.05)
    return stats

ramp_stats_el   = compute_ramp_stats(residual_el_ts,   "electricity")
ramp_stats_h2   = compute_ramp_stats(residual_h2_ts,   "hydrogen")
ramp_stats_heat = compute_ramp_stats(residual_heat_ts, "heat")
all_ramp_stats  = {**ramp_stats_el, **ramp_stats_h2, **ramp_stats_heat}

print("\n" + "=" * 70)
print("  RAMPING METRICS (Residual Load Changes)")
print("=" * 70)
for key, val in all_ramp_stats.items():
    print(f"  {key:<45} {val:>10.2f} MW/h")
print("=" * 70)

# ── 2. Storage utilisation metrics ────────────────────────────────────────────
# Load-shift ratio: total energy cycled / installed capacity
# Higher = storage is used more intensively relative to its size
# Also compute: equivalent full cycles per year

def storage_metrics(charge_ts, discharge_ts, soc_ts, capacity, label):
    """
    Compute storage flexibility metrics.
    
    Metrics:
    - load_shift_ratio: total energy shifted / capacity (dimensionless)
    - full_cycles_per_year: equivalent number of full charge-discharge cycles
    - avg_soc_fraction: average SoC as fraction of capacity
    - utilisation_fraction: fraction of hours storage is actively charging/discharging
    - max_soc_fraction: peak SoC reached as fraction of capacity
    """
    if capacity <= 0:
        return {f"{label}_{k}": 0.0 for k in
                ["load_shift_ratio", "full_cycles_yr", "avg_soc_frac",
                 "utilisation_frac", "max_soc_frac", "energy_charged_mwh",
                 "energy_discharged_mwh"]}

    total_charged    = charge_ts.sum()
    total_discharged = discharge_ts.sum()
    total_cycled     = (total_charged + total_discharged) / 2  # count each MWh once

    active_hours = ((charge_ts > 0.01) | (discharge_ts > 0.01)).sum() #NOTE: "|" is bitwise OR for pandas Series

    return {
        f"{label}_load_shift_ratio":      total_cycled / capacity,
        f"{label}_full_cycles_yr":        total_discharged / capacity, #FIXME: should be total_charged?
        f"{label}_avg_soc_frac":          (soc_ts / capacity).mean(),
        f"{label}_utilisation_frac":      active_hours / len(charge_ts),
        f"{label}_max_soc_frac":          (soc_ts / capacity).max(),
        f"{label}_energy_charged_mwh":    total_charged,
        f"{label}_energy_discharged_mwh": total_discharged,
    }

storage_met_bat   = storage_metrics(bat_charge, bat_discharge, battery_soc,
                                     capacity_battery, "battery")
storage_met_sc    = storage_metrics(saltcavern_charge, saltcavern_discharge,
                                     saltcavern_soc, capacity_saltcavern, "saltcavern")
storage_met_ts    = storage_metrics(thermalstorage_charge, thermalstorage_discharge,
                                     thermalstorage_soc, capacity_thermalstorage,
                                     "thermalstorage")
all_storage_met   = {**storage_met_bat, **storage_met_sc, **storage_met_ts}

print("\n" + "=" * 65)
print("  STORAGE FLEXIBILITY METRICS")
print("=" * 65)
for key, val in all_storage_met.items():
    print(f"  {key:<45} {val:>10.3f}")
print("=" * 65)

# ── 3. Sector coupling exchange ratios ────────────────────────────────────────
# These ratios quantify how much cross-sector conversion is happening
# relative to total production/demand in each carrier.
# Methodology follows PATHFNDR energy exchange ratio definition.

total_res_gen    = pv_flow.sum() + wind_flow.sum()
total_el_demand  = el_demand_ts.sum()
total_h2_demand  = h2_demand_ts.sum()
total_heat_demand= heat_demand_ts.sum()

# Power-to-X ratios (electricity converted to other carriers)
ptx_el_to_h2     = el_input.sum()       / total_res_gen   if total_res_gen   > 0 else 0.0
ptx_el_to_heat   = heatpump_input.sum() / total_res_gen   if total_res_gen   > 0 else 0.0
ptx_el_total     = (el_input.sum() + heatpump_input.sum()) / total_res_gen if total_res_gen > 0 else 0.0

# Power-from-X ratios (other carriers converted back to electricity)
pfx_h2_to_el     = chp_output_el.sum()  / total_el_demand if total_el_demand > 0 else 0.0

# Sector coupling depth: fraction of each demand met via cross-sector conversion
sc_h2_from_el    = el_output.sum()       / total_h2_demand   if total_h2_demand   > 0 else 0.0
sc_heat_from_el  = heatpump_output.sum() / total_heat_demand if total_heat_demand > 0 else 0.0
sc_heat_from_h2  = chp_output_heat.sum() / total_heat_demand if total_heat_demand > 0 else 0.0
sc_el_from_h2    = chp_output_el.sum()   / total_el_demand   if total_el_demand   > 0 else 0.0

# Electrolyser flexibility: what fraction of its input is used for H2 vs heat
el_flexibility_ratio = el_output.sum() / (el_output.sum() + el_output_heat.sum()) if (el_output.sum() + el_output_heat.sum()) > 0 else 0.0

print("\n" + "=" * 65)
print("  SECTOR COUPLING METRICS")
print("=" * 65)
print(f"  {'Power-to-H2 ratio (of RES gen)':<45} {ptx_el_to_h2:>8.3f}")
print(f"  {'Power-to-Heat ratio (of RES gen)':<45} {ptx_el_to_heat:>8.3f}")
print(f"  {'Total Power-to-X ratio (of RES gen)':<45} {ptx_el_total:>8.3f}")
print(f"  {'Power-from-H2 ratio (of el demand)':<45} {pfx_h2_to_el:>8.3f}")
print(f"  {'H2 demand met by electrolyser':<45} {sc_h2_from_el:>8.3f}")
print(f"  {'Heat demand met by heat pump':<45} {sc_heat_from_el:>8.3f}")
print(f"  {'Heat demand met by CHP':<45} {sc_heat_from_h2:>8.3f}")
print(f"  {'El demand met by CHP':<45} {sc_el_from_h2:>8.3f}")
print(f"  {'Electrolyser H2 output fraction':<45} {el_flexibility_ratio:>8.3f}")
print("=" * 65)

# ── 4. Self-sufficiency and curtailment metrics ────────────────────────────────
# SSR: fraction of demand met without curtailment (proxy for system reliability)
# SCR: fraction of generated energy that was actually used

def ssr_scr(demand_ts, curtailment_ts, generation_ts, label):
    """
    Self-Sufficiency Ratio and Self-Consumption Ratio.
    
    SSR = min(demand_ts, generation_ts)/demand_ts if demand_ts > 0 else 1.0
    SCR = min(demand_ts, generation_ts) / generation_ts if generation_ts > 0 else 0.0
    """
    total_gen  = generation_ts.sum()
    total_dem  = demand_ts.sum()
    total_curt = curtailment_ts.sum()
    #FIXME: Pastore et al. (2026) define SCR = min(el_demand_ts, pv_flow + wind_flow)/(pv_flow + wind_flow) for the electricity bus, which is a more direct measure of how much of the variable RES generation is actually used to meet demand.
    #TODO: update formula considering electorlyser + HP as loads, so:
    #  SCR_el (per timestep) = min(el_demand_ts + el_input + heatpump_input + bat_charge, pv_flow + wind_flow) / (pv_flow + wind_flow)
    #  SCR_el_total = sum(min(el_demand_ts + el_input + heatpump_input + bat_charge, pv_flow + wind_flow)) / sum(pv_flow + wind_flow)
    #  SSR_el (per timestep) = min(el_demand_ts + el_input + heatpump_input + bat_charge, pv_flow + wind_flow) / el_demand_ts
    #TODO: Have a plot with SCR (and SSR) vs (pv_flow + wind_flow)/el_demand_ts to see how they relate to RES penetration and demand levels.
    scr = (total_gen - total_curt) / total_gen if total_gen > 0 else 0.0
    # In a fully-served system, SSR = 1.0 by construction (all demand met).
    # We instead report the curtailment fraction as a flexibility stress indicator.
    curtailment_frac = total_curt / total_gen if total_gen > 0 else 0.0

    print(f"  {label:<20} SCR={scr:.3f}  curtailment_frac={curtailment_frac:.3f}  "
          f"curtailed={total_curt:.0f} MWh  generated={total_gen:.0f} MWh")
    return scr, curtailment_frac

print("\n" + "=" * 75)
print("  SELF-CONSUMPTION & CURTAILMENT")
print("=" * 75)
scr_el,   curt_frac_el   = ssr_scr(el_demand_ts,   curtailment_el,   pv_flow + wind_flow + chp_output_el,   "Electricity")
scr_h2,   curt_frac_h2   = ssr_scr(h2_demand_ts,   curtailment_h2,   el_output,                              "Hydrogen")
scr_heat, curt_frac_heat = ssr_scr(heat_demand_ts, curtailment_heat, heatpump_output + chp_output_heat + el_output_heat, "Heat")
print("=" * 75)

# ── 5. Flexibility provision share (per bus, per source) ──────────────────────
# What fraction of demand on each bus was served by each flexibility provider?
# This gives a "flexibility stack" decomposition.

def flex_provision_shares(label, demand_total, components: dict):
    """
    Print and return the share of total demand met by each flexibility source.
    """
    print(f"\n  {label} bus flexibility provision shares:")
    shares = {}
    for name, energy in components.items():
        share = energy / demand_total if demand_total > 0 else 0.0
        shares[name] = share
        print(f"    {name:<35} {energy:>10.0f} MWh  ({share*100:>6.2f}%)")
    return shares

print("\n" + "=" * 65)
print("  FLEXIBILITY PROVISION SHARES")
print("=" * 65)

flex_shares_el = flex_provision_shares(
    "Electricity", total_el_demand,
    {
        "PV (direct)":               pv_flow.sum(), #FIXME: PV and wind should be system-wide provision, not just direct to electricity demand
        "Wind (direct)":             wind_flow.sum(),
        "CHP (H2→El)":               chp_output_el.sum(),
        "Battery discharge":         bat_discharge.sum(),
        "Curtailment (unserved)":    curtailment_el.sum(),
    }
)

flex_shares_h2 = flex_provision_shares(
    "Hydrogen", total_h2_demand,
    {
        "Electrolyser (El→H2)":      el_output.sum(),
        "Salt cavern discharge":     saltcavern_discharge.sum(),
        "Curtailment (unserved)":    curtailment_h2.sum(),
    }
)

flex_shares_heat = flex_provision_shares(
    "Heat", total_heat_demand,
    {
        "Heat Pump (El→Heat)":       heatpump_output.sum(),
        "CHP (H2→Heat)":  chp_output_heat.sum(),
        "Electrolyser waste heat":   el_output_heat.sum(),
        "Thermal storage discharge": thermalstorage_discharge.sum(),
        "Curtailment (unserved)":    curtailment_heat.sum(),
    }
)
print("=" * 65)


# =============================================================================
# FLEXIBILITY VISUALISATIONS
# =============================================================================

# ── FLEX PLOT 1 — Residual load duration curves (all 3 buses) ─────────────────
# The residual load duration curve shows how many hours the system
# is in surplus vs deficit — the area above zero requires upward flexibility,
# the area below zero requires downward flexibility or curtailment.

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, rl_ts, label, color in zip(
    axes,
    [residual_el_ts, residual_h2_ts, residual_heat_ts],
    ["Electricity", "Hydrogen", "Heat"],
    ["darkorange", "steelblue", "firebrick"],
):
    sorted_rl = rl_ts.sort_values(ascending=False).values
    hours     = np.arange(len(sorted_rl))

    # colour above/below zero differently
    ax.fill_between(hours, sorted_rl,
                    where=(sorted_rl >= 0), color=color,    alpha=0.6,
                    label="Deficit (upward flex needed)")
    ax.fill_between(hours, sorted_rl,
                    where=(sorted_rl <  0), color="green",  alpha=0.5,
                    label="Surplus (downward flex / storage)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"{label} Residual Load\nDuration Curve")
    ax.set_xlabel("Hours (ranked)")
    ax.set_ylabel("Net load / [MW]")
    ax.legend(fontsize=8)

plt.suptitle("Residual Load Duration Curves — Flexibility Stress by Bus",
             fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX1_1residual_load_duration_curves")
plt.show()


# ── FLEX PLOT 2 — Ramping distribution (violin + box, all buses) ──────────────
# The ramp distribution reveals the "speed" of flexibility needed.
# Wide distributions = high ramping requirements → need fast-response assets.
#TODO: Create Violin plots showing residual loads and curtailment distributions for each bus

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


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, rl_ts, label in zip(
    axes,
    [residual_el_ts, residual_h2_ts, residual_heat_ts],
    ["Electricity", "Hydrogen", "Heat"],
):
    ramp_data = {}
    for w in [1, 3, 8]:
        ramp_data[f"{w}h ramp"] = rl_ts.diff(w).values  #This is absolute change in net load over the window, not normalised by time, so units are MW/h
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
func.savefigure(fig, "FLEX1_2ramping_distributions")
plt.show()


# ── FLEX PLOT 3 — Monthly flexibility stack (stacked bar by source) ────────────
# Shows how each flexibility source contributes per month.
# Allows seeing seasonal patterns in flexibility reliance.

monthly_flex_el = pd.DataFrame({
    "PV":               pv_flow.values,#TODO: Subtract electrolyser and HP consumption
    "Wind":             wind_flow.values,
    "CHP (H2→El)":      chp_output_el.values,
    "Battery":          bat_discharge.values,
}, index=idx).resample("ME").sum()

monthly_flex_h2 = pd.DataFrame({
    "Electrolyser":     el_output.values, #TODO: have it as electrolyser H2 output minus CHP H2 consumption
    "Salt Cavern":      saltcavern_discharge.values,
}, index=idx).resample("ME").sum()

monthly_flex_heat = pd.DataFrame({
    "Heat Pump":        heatpump_output.values,
    "CHP heat":         chp_output_heat.values,
    "Electrolyser heat":el_output_heat.values,
    "Thermal Storage":  thermalstorage_discharge.values,
}, index=idx).resample("ME").sum()

month_labels = [d.strftime("%b") for d in monthly_flex_el.index]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for ax, df_monthly, demand_monthly, label in zip(
    axes,
    [monthly_flex_el, monthly_flex_h2, monthly_flex_heat],
    [
        pd.Series(el_demand_ts.values, index=idx).resample("ME").sum(),
        pd.Series(h2_demand_ts.values, index=idx).resample("ME").sum(),
        pd.Series(heat_demand_ts.values, index=idx).resample("ME").sum(),
    ],
    ["Electricity", "Hydrogen", "Heat"],
):
    df_monthly.plot(kind="bar", ax=ax, stacked=True, width=0.8, alpha=0.85)
    ax.plot(range(len(demand_monthly)), demand_monthly.values,
            "k--o", lw=1.5, ms=4, label="Demand", zorder=5)
    ax.set_title(f"{label} Bus\nMonthly Flexibility Stack")
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel("Energy / [MWh]")
    ax.legend(fontsize=7, loc="upper right")

#plt.suptitle("Monthly Flexibility Provision by Source and Bus", fontsize=13)
plt.suptitle("What goes into each bus vs end-use", fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX1_3monthly_flexibility_stack")
plt.show()


# ── FLEX PLOT 4 — Sector coupling Sankey-style bar chart ──────────────────────
# A horizontal bar chart showing energy flows between sectors.
# Visualises how much electricity goes to each conversion pathway.

fig, ax = plt.subplots(figsize=(10, 6))

categories = [
    "El → Electrolyser\n(El→H2)",
    "El → Heat Pump\n(El→Heat)",
    "El → Battery charge",
    "H2 → CHP (El out)",
    "H2 → CHP (Heat out)",
    "H2 → Salt cavern",
]
values = [
    el_input.sum(),
    heatpump_input.sum(),
    bat_charge.sum(),
    chp_output_el.sum(),
    chp_output_heat.sum(),
    saltcavern_charge.sum(),
]
colors = [
    "steelblue", "steelblue", "steelblue",
    "firebrick", "firebrick",
    "purple",
]

bars = ax.barh(categories, values, color=colors, alpha=0.75, edgecolor="white")
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f} MWh", va="center", fontsize=9)

ax.set_xlabel("Annual Energy Flow / [MWh]")
ax.set_title("Annual Cross-Sector Energy Flows\n(Sector Coupling Magnitude)")
ax.axvline(0, color="black", lw=0.8)
plt.tight_layout()
func.savefigure(fig, "FLEX1_4sector_coupling_flows")
plt.show()


# ── FLEX PLOT 5 — Storage SOC heatmaps (hour of day × month) ──────────────────
# Shows WHEN each storage is used during the year.
# Reveals daily vs seasonal flexibility patterns.

def soc_heatmap(soc_ts, capacity, label, ax, cmap="YlOrRd"):
    """Plot a heatmap of SoC (normalised 0-1) by hour-of-day and month."""
    if capacity <= 0:
        ax.set_visible(False)
        return
    soc_norm = (soc_ts / capacity).copy()
    soc_norm.index = pd.DatetimeIndex(soc_norm.index)
    df_hm = soc_norm.to_frame("soc")
    df_hm["month"]  = df_hm.index.month
    df_hm["hour"]   = df_hm.index.hour
    pivot = df_hm.pivot_table(values="soc", index="hour", columns="month", aggfunc="mean")
    pivot.columns = [pd.Timestamp(2023, m, 1).strftime("%b") for m in pivot.columns]
    sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Mean SoC / [fraction of capacity]"})
    ax.set_title(f"{label}\n(mean SoC by hour & month)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
soc_heatmap(battery_soc,         capacity_battery,         "Battery",          axes[0], cmap="Blues")
soc_heatmap(saltcavern_soc,      capacity_saltcavern,      "Salt Cavern",       axes[1], cmap="Oranges")
soc_heatmap(thermalstorage_soc,  capacity_thermalstorage,  "Thermal Storage",  axes[2], cmap="Reds")

plt.suptitle("Storage Utilisation Patterns — Mean SoC by Hour of Day and Month",
             fontsize=13)
plt.tight_layout()
func.savefigure(fig, "FLEX1_5storage_soc_heatmaps")
plt.show()


# ── FLEX PLOT 6 — Quantitative flexibility comparison radar chart ──────────────
# Normalised spider/radar chart comparing all flexibility types
# on a common 0–1 scale. Useful for scenario comparison.

# Define metrics (label, value, max_reference)
# max_reference is the theoretical maximum (e.g., total generation for fractions)
flex_radar_metrics = {
    "PtX ratio\n(el→H2+heat)":      ptx_el_total,
    "CHP back-coupling\n(H2→el)":   pfx_h2_to_el,
    "Battery\nutilisation":          all_storage_met.get("battery_utilisation_frac", 0),
    "Salt Cavern\nutilisation":      all_storage_met.get("saltcavern_utilisation_frac", 0),
    "Thermal Storage\nutilisation":  all_storage_met.get("thermalstorage_utilisation_frac", 0),
    "El SCR\n(1-curtailment)":       scr_el,
    "H2 SCR\n(1-curtailment)":       scr_h2,
    "Heat SCR\n(1-curtailment)":     scr_heat,
}#TODO: Add SSR

labels_r  = list(flex_radar_metrics.keys())
values_r  = [min(v, 1.0) for v in flex_radar_metrics.values()]  # clip to [0,1]
N         = len(labels_r)
angles    = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values_r += values_r[:1]   # close the polygon
angles   += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
ax.plot(angles, values_r, "o-", lw=2, color="steelblue")
ax.fill(angles, values_r, alpha=0.25, color="steelblue")
ax.set_thetagrids(np.degrees(angles[:-1]), labels_r, fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
ax.set_title("Flexibility Profile — Normalised Radar Chart\n"
             "(1.0 = maximum possible for each metric)", fontsize=11, pad=20)
plt.tight_layout()
func.savefigure(fig, "FLEX1_6radar_chart")
plt.show()


# ── FLEX PLOT 7 — Flexibility type decomposition (annual, stacked bar) ─────────
# Compares four flexibility types by their annual energy contribution.
# Useful for a single-number comparison table across scenarios.

flex_types = {
    "Supply-side\n(RES)":        pv_flow.sum() + wind_flow.sum(),
    "Storage\n(Battery)":        bat_discharge.sum(),
    "Storage\n(Salt Cavern)":    saltcavern_discharge.sum(),
    "Storage\n(Thermal)":        thermalstorage_discharge.sum(),
    "Sector coupling\n(El→H2)":  el_output.sum(),
    "Sector coupling\n(El→Heat)":heatpump_output.sum(),
    "Sector coupling\n(El→Heat)":  el_output_heat.sum(),
    "Sector coupling\n(H2→El)":  chp_output_el.sum(),
    "Sector coupling\n(H2→Heat)":chp_output_heat.sum(),
}

fig, ax = plt.subplots(figsize=(12, 5))
colors_bar = plt.cm.tab10(np.linspace(0, 1, len(flex_types)))
bars = ax.bar(list(flex_types.keys()), list(flex_types.values()),
              color=colors_bar, edgecolor="white", alpha=0.85)
for bar, val in zip(bars, flex_types.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(flex_types.values()) * 0.01,
            f"{val/1e3:.1f} GWh", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Annual Energy / [MWh]")
ax.set_title("Annual Flexibility Provision by Type\n"
             "(Energy delivered by each flexibility source)")
ax.set_ylim(0, max(flex_types.values()) * 1.12)
plt.tight_layout()
func.savefigure(fig, "FLEX1_7annual_flexibility_decomposition")
plt.show()


# ── FLEX PLOT 8 — Rolling flexibility index (weekly rolling window) ────────────
# Shows how the flexibility burden shifts throughout the year.
# Computed as the fraction of electricity demand met by non-RES sources
# (storage + CHP), rolling 7-day window.
#TODO: think about flex_index_ variables' defintion. DO they make sense? Do they capture the intended concept of "flexibility reliance"?
window = 7*24  # 7 days × 24 h

non_res_el = bat_discharge + chp_output_el  # non-RES contributions to el bus
flex_index_el = (non_res_el / el_demand_ts.replace(0, np.nan)).rolling(window).mean()

non_res_h2    = saltcavern_discharge  # non-electrolyser H2
flex_index_h2 = (non_res_h2 / h2_demand_ts.replace(0, np.nan)).rolling(window).mean()

non_res_heat  = thermalstorage_discharge + el_output_heat
flex_index_heat = (non_res_heat / heat_demand_ts.replace(0, np.nan)).rolling(window).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(flex_index_el.index,   flex_index_el.values,   label="Electricity bus",  color="darkorange",  lw=1.2)
ax.plot(flex_index_h2.index,   flex_index_h2.values,   label="Hydrogen bus",     color="steelblue",   lw=1.2)
ax.plot(flex_index_heat.index, flex_index_heat.values, label="Heat bus",         color="firebrick",   lw=1.2)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Fraction of demand met by non-primary sources, 7d avg)")
ax.set_xlabel("Date")
ax.set_title("Rolling Flexibility Index — Seasonal Dynamics of Flexibility Reliance")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend()
plt.tight_layout()
func.savefigure(fig, "FLEX1_8rolling_flexibility_index")
plt.show()


# =============================================================================
# FLEXIBILITY SUMMARY TABLE (for logbook / scenario comparison)
# =============================================================================

flex_summary = {
    # ── Ramping ───────────────────────────────────────────────────────────────
    "ramp_el_1h_max_up_mw":        ramp_stats_el.get("electricity_ramp_1h_max_up",   0),
    "ramp_el_1h_max_down_mw":      ramp_stats_el.get("electricity_ramp_1h_max_down", 0),
    "ramp_el_1h_p95_up_mw":        ramp_stats_el.get("electricity_ramp_1h_p95_up",   0),
    "ramp_h2_1h_max_up_mw":        ramp_stats_h2.get("hydrogen_ramp_1h_max_up",      0),
    "ramp_heat_1h_max_up_mw":      ramp_stats_heat.get("heat_ramp_1h_max_up",        0),

    # ── Storage ───────────────────────────────────────────────────────────────
    "battery_load_shift_ratio":    all_storage_met.get("battery_load_shift_ratio",        0),
    "battery_full_cycles_yr":      all_storage_met.get("battery_full_cycles_yr",          0),
    "saltcavern_load_shift_ratio": all_storage_met.get("saltcavern_load_shift_ratio",     0),
    "saltcavern_full_cycles_yr":   all_storage_met.get("saltcavern_full_cycles_yr",       0),
    "thermalstorage_load_shift":   all_storage_met.get("thermalstorage_load_shift_ratio", 0),

    # ── Sector coupling ───────────────────────────────────────────────────────
    "ptx_el_to_h2_ratio":          float(ptx_el_to_h2),
    "ptx_el_to_heat_ratio":        float(ptx_el_to_heat),
    "ptx_total_ratio":             float(ptx_el_total),
    "pfx_h2_to_el_ratio":          float(pfx_h2_to_el),
    "sc_h2_from_el_fraction":      float(sc_h2_from_el),
    "sc_heat_from_el_fraction":    float(sc_heat_from_el),
    "sc_heat_from_h2_fraction":    float(sc_heat_from_h2),
    "sc_el_from_h2_fraction":      float(sc_el_from_h2),

    # ── Self-consumption ──────────────────────────────────────────────────────
    "scr_electricity":             float(scr_el),
    "scr_hydrogen":                float(scr_h2),
    "scr_heat":                    float(scr_heat),
    "curtailment_frac_el":         float(curt_frac_el),
    "curtailment_frac_h2":         float(curt_frac_h2),
    "curtailment_frac_heat":       float(curt_frac_heat),
}

print("\n" + "=" * 65)
print("  FLEXIBILITY SUMMARY (for logbook)")
print("=" * 65)
for k, v in flex_summary.items():
    print(f"  {k:<45} {v:>10.4f}")
print("=" * 65)



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

logbook["flexibility"] = flex_summary

func.save_logbook_csv(logbook, DATA_DIR, SCENARIO_NAME)