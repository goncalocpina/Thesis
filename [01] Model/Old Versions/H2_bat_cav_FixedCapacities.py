'''Author: Gonçalo Costa Pina
Date_Created: 2026-01-30 (30th January 2026)
Date_Modified: 2026-03-27

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
from Parameters import parameters as par
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import numpy_financial as npf   # pip install numpy-financial
from collections import defaultdict
import funcs as func

#=============================================================================
# SCENARIO DEFINITION
#=============================================================================
#Scenario name is structured as "{Part of system}_{Unit being analysed}_{Parameters varied}"
SCENARIO_NAME = "H2ONLY_WIND_CF_CAPACITY"   # ← change per run



# ============================================================================
# IMPORTING PROFILES
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent   # ← point to your directory
data_path = (DATA_DIR.parent
    / "[02] Data"
    / "Supply")

pv_file = data_path / "ninja_pv_51.9244_4.4778_corrected.csv"
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

 
# ── H2 demand ───────────────────────────────────────────────────
# (absolute, MWh per hour, 8760 rows)
demand_file = (
    DATA_DIR.parent
    / "[02] Data"
    / "Demand"
    / "Demand_Profiles.csv")

df_h2 = pd.read_csv(demand_file, sep=";")
df_h2["Datetime (UTC)"] = pd.to_datetime(df_h2["Datetime (UTC)"], format="%d/%m/%Y %H:%M")
df_h2 = df_h2[["Datetime (UTC)", "Rotterdam_total_gas_demand [MWh]"]].copy()
df_h2["Rotterdam_total_gas_demand [MWh]"] = pd.to_numeric(df_h2["Rotterdam_total_gas_demand [MWh]"], errors="coerce")
df_h2 = df_h2.dropna(subset=["Rotterdam_total_gas_demand [MWh]"])
df_h2 = df_h2.sort_values("Datetime (UTC)").reset_index(drop=True)
demand_h2 = df_h2["Rotterdam_total_gas_demand [MWh]"].values
print("Demand profile loaded:", len(df_h2), "rows")

CF_PV_BASELINE = df_pv["electricity"].sum()/len(df_pv["electricity"]) 
CF_WIND_BASELINE = df_wind["electricity"].sum()/len(df_wind["electricity"])

#CFs TO TEST. Will be used to scale the input profiles
#To TEST DIFFERESNT CFs, CHANGE NUMERATOR ONLY
CF_PV = CF_PV_BASELINE /CF_PV_BASELINE         
CF_WIND = CF_WIND_BASELINE /CF_WIND_BASELINE


timeseries = pd.DataFrame({
    "PV": df_pv["electricity"].values * CF_PV,
    "Wind": df_wind["electricity"].values * CF_WIND,
    "H2_Demand": demand_h2
}, index=df_h2["Datetime (UTC)"])

# Fix the index
timeseries.index = pd.DatetimeIndex(timeseries.index).round("h")
timeseries.index.name = "time"
timeseries = timeseries.asfreq("h")


'''print("\nTimeseries DataFrame created successfully:")
print(timeseries.iloc[0:15])
print("\nTimeseries DataFrame info:")
timeseries.info()
print(timeseries.index.freq)'''


# =============================================================================
# OEMOF MODEL SETUP
# =============================================================================

# Create energy system
energysystem = solph.EnergySystem(timeindex=timeseries.index, infer_last_interval=True)

# Create buses
bus_electricity = solph.Bus(label='electricity')
bus_H2 = solph.Bus(label='hydrogen')

# Create components 
# ── PV ───────────────────────────────────────────────────
PV_OPTIMAL = 453.2 #MW
pv = solph.components.Source(
    label="pv",
    outputs={bus_electricity: solph.Flow(
        fix=timeseries["PV"],
        nominal_value=solph.Investment(
                ep_costs=(
                    economics.annuity(capex = par.PV_CAPEX_PANELS, n=par.PV_LIFETIME_SYSTEM, wacc=par.WACC) 
                    + economics.annuity(capex=par.PV_CAPEX_INVERTER, n= par.PV_LIFETIME_SYSTEM, wacc = par.WACC, u = par.PV_LIFETIME_INVERTER)
                    + par.PV_OPEX), # annualized CAPEX in €/MW/year and OPEX (same units)
                existing=0
            ),
        variable_costs=0  
        )
    },)

# ── Wind ───────────────────────────────────────────────────
WIND_OPTIMAL =1967.1 #MW
scaling =0
wind = solph.components.Source(
    label="wind",
    outputs={bus_electricity: solph.Flow(
        fix=timeseries["Wind"],
        nominal_value=solph.Investment(
                ep_costs=economics.annuity(capex=par.WIND_CAPEX, n=par.WIND_LIFETIME_SYSTEM, wacc=par.WACC) + par.WIND_OPEX,  # annualized CAPEX in €/MW/year and OPEX (same units)
                #ep_costs=par.WIND_CAPEX/par.WIND_LIFETIME_SYSTEM + par.WIND_OPEX,  
                existing=0,
            ),
        variable_costs=0 #OPEX isn't here because it is a fixed cost for this unit, given in €/MW/year 
        )
    },)

# ── Electrolyser ───────────────────────────────────────────────────
electrolyser = solph.components.Converter(
    label="electrolyser",
    inputs={bus_electricity: solph.Flow(
        nominal_value=solph.Investment(
            ep_costs=(
                    economics.annuity(capex = par.ELECTROLYSER_CAPEX_SYSTEM, n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC)
                    + economics.annuity(capex = par.ELECTROLYSER_CAPEX_STACK, n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC, u=par.ELECTROLYSER_LIFETIME_STACK)
                    + par.ELECTROLYSER_OPEX ),  # annualized CAPEX for system and stack
            minimum=0,
            existing=0,
        ),
        min=par.ELECTROLYSER_MIN_LOAD,  # TODO: Requires unit commitment logic; disabled for now
        variable_costs=0, #€/MWh
    )},
    outputs={bus_H2: solph.Flow()}, #, bus_heat:solph.Flow()
    conversion_factors={bus_H2: par.ELECTROLYSER_EFFICIENCY}, #, bus_heat: par.ELECTROLYSER_RECOVERABLE_HEAT
)

# ── Battery ───────────────────────────────────────────────────
battery = solph.components.GenericStorage(
    label="battery",
    inputs={
        bus_electricity: solph.Flow(
            nominal_value=solph.Investment(
                #par.BATTERY_CAPEX_POWER = 0. Defined simply for consistency
                ep_costs=economics.annuity(
                    capex=par.BATTERY_CAPEX_POWER, 
                    n=par.LIFETIME, wacc=par.WACC, 
                    u = par.BATTERY_LIFETIME_SYSTEM) 
                +par.BATTERY_OPEX_FIX,          # €/MW/yr → on power capacity
            ),
            variable_costs=par.BATTERY_OPEX_VAR / 2,   # €/MWh throughput
        )
    },
    outputs={
        bus_electricity: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=0,                              # power cost already on input
            ),
            variable_costs=par.BATTERY_OPEX_VAR / 2,   # €/MWh throughput
        )
    },
    nominal_storage_capacity=solph.Investment(
        ep_costs=(
            economics.annuity(
                capex=par.BATTERY_CAPEX_SYSTEM,          
                n=par.LIFETIME,
                wacc=par.WACC,
                u=par.BATTERY_LIFETIME_SYSTEM,             
            )
            + economics.annuity(
                capex=par.BATTERY_CAPEX_BATTERY_PACK,   # pack replaced every 5 years
                n=par.LIFETIME ,
                wacc=par.WACC,
                u=par.BATTERY_LIFETIME_BATTERY_PACK,
            )
            # no OPEX here because no OPEX compoenent is in €/MWh/yr, but rather €/MW/yr or €/MWh
        ),
        minimum=0,
        existing=0,
    ),
    invest_relation_input_capacity=1 / par.BATTERY_MIN_CHARGE_TIME,
    invest_relation_output_capacity=1 / par.BATTERY_MIN_DISCHARGE_TIME,
    inflow_conversion_factor=par.BATTERY_EFFICIENCY_CHARGE,
    outflow_conversion_factor=par.BATTERY_EFFICIENCY_DISCHARGE,
    loss_rate=par.BATTERY_SELF_DISCHARGE_RATE,
    initial_storage_level=0,
)

# ── Salt Cavern ───────────────────────────────────────────────────
saltcavern = solph.components.GenericStorage(
    label="saltcavern",
    inputs={
        bus_H2: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=(
                    economics.annuity(
                        capex=par.SALTCAVERN_CAPEX_COMPRESSOR,   # €/MW, with replacement
                        n=par.SALTCAVERN_LIFETIME,
                        wacc=par.WACC,
                        u=par.SALTCAVERN_LIFETIME_COMPRESSOR,    # replaced at year 15
                    )
                    # no fixed OPEX in €/MW/yr for compressor
                ),
                minimum=0,
                existing=0,
            ),
            variable_costs=par.SALTCAVERN_OPEX_COMPRESSOR,      # €/MWh compressed
        )
    },
    outputs={
        bus_H2: solph.Flow(
            nominal_value=solph.Investment(
                ep_costs=0,                                      # compressor cost already on input
            ),
        )
    },
    nominal_storage_capacity=solph.Investment(
        ep_costs=(
            economics.annuity(
                capex=par.SALTCAVERN_CAPEX,                      # €/MWh, geological site
                n=par.SALTCAVERN_LIFETIME,
                wacc=par.WACC,
            )
            + economics.annuity(
                capex=par.SALTCAVERN_CAPEX_CUSHION_GAS,          # €/MWh, locked-in cushion gas
                n=par.SALTCAVERN_LIFETIME,
                wacc=par.WACC,
                # no u — cushion gas is never replaced
            )
            + par.SALTCAVERN_OPEX                                # €/MWh/yr, cavern O&M
        ),
        minimum=0,
        existing=0,
    ),
    # ── C-rate: no fixed ratio imposed — compressor and cavern sized independently
    invest_relation_input_capacity=None,
    invest_relation_output_capacity=None,
    # ── Efficiency ───────────────────────────────────────────────────
    inflow_conversion_factor=1,
    outflow_conversion_factor=par.SALTCAVERN_EFFICIENCY,         # = 1
    # ── Self-discharge ───────────────────────────────────────────────
    loss_rate=par.SALTCAVERN_SELF_DISCHARGE_RATE,                # = 0
    # ── Cushion gas — minimum SOC ────────────────────────────────────
    min_storage_level=par.SALTCAVERN_CUSHIONGAS_FRACTION,        # = 0.3
    # ── Volume loss — NOT supported in investment mode ───────────────
    # fixed_losses_relative=par.SALTCAVERN_RELATIVE_VOLUME_LOSS / 8760
    # This parameter is unsupported in investment mode per GenericStorage docs.
    # Workaround: reduce Investment maximum by expected capacity loss if needed.
    initial_storage_level=par.SALTCAVERN_CUSHIONGAS_FRACTION,    # start at minimum SOC
)

# ── H2 Demand ───────────────────────────────────────────────────
H2Demand = solph.components.Sink(
    label="H2_demand",
    inputs={
        bus_H2: solph.Flow(
            fix=timeseries["H2_Demand"],
            nominal_value=1,  # No investment decision for demand, so nominal_value is set to 1
            variable_costs=0  # No variable cost for demand
        )
    }
)


# Sink for excess electricity (exports, curtailment, or other uses)
# Prevents infeasibility by allowing excess renewable production to be disposed of
electricity_curtailement = solph.components.Sink(
    label="electricity_curtailement",
    inputs={
        bus_electricity: solph.Flow(
            variable_costs=0  # Free to curtail (or use a small cost if preferred)
        )
    }
)


print("Max H2 demand (MWh/h):", timeseries["H2_Demand"].max())
print("Annual H2 demand (MWh):", timeseries["H2_Demand"].sum())


# Add components to energy system
energysystem.add(bus_electricity, bus_H2, pv, wind, electrolyser, H2Demand, battery, saltcavern, electricity_curtailement)

model = solph.Model(energysystem)
model.solve(
    solver='cbc',
    solve_kwargs={'tee': True}  # Show progress in console
)


print("Processing results...\n")

results = solph.processing.results(model)
string_results = views.convert_keys_to_strings(results)

# See all keys (all flows in the model)
# ===== GET OPTIMAL CAPACITIES =====

# PV
capacity_pv = results[(pv, bus_electricity)]['scalars']['invest']
capacity_wind = results[(wind, bus_electricity)]['scalars']['invest']
capacity_electrolyser = results[(bus_electricity, electrolyser)]['scalars']['invest']
capacity_battery = results[(battery, None)]['scalars']['invest']
capacity_saltcavern = results[(saltcavern, None)]['scalars']['invest']


# See all keys (edges) in the results dict
for key in results[(bus_electricity, electrolyser)].keys():
    print("Result key: ", key)

pv_results = solph.views.node(results, "pv")
print(pv_results["sequences"].head())
print(pv_results["sequences"].columns.tolist())  # see what's available
#IMPORTANT CHECK, as it allows to ensure electrolyser efficiency is well applied
electro_results = solph.views.node(results, "electrolyser") 
print(electro_results["sequences"].head())
print(electro_results["sequences"].columns.tolist()) 


print("="*90)
print("🎯 OPTIMISATION RESULTS")
print("="*90)
print("\n💡 OPTIMAL CAPACITIES:")
print(f"  PV: {capacity_pv:.1f} MW")
print(f"  Wind: {capacity_wind:.1f} MW")
print(f"  Electrolyser (Input): {capacity_electrolyser:.1f} MW")
print(f"  TOTAL RES CAPACITY INSTALLED: {capacity_pv + capacity_wind:.1f} MW")
print(f"  Battery: {capacity_battery:.1f} MWh")
print(f"  Salt Cavern: {capacity_saltcavern:.1f} MWh")



# =============================================================================
# VISUALISATION
# =============================================================================
print(results[(battery, None)]['sequences'].columns)
print(results[(saltcavern, None)]['sequences'].columns)
# ── Extract timeseries from results ──────────────────────────────────────────

pv_flow       = results[(pv, bus_electricity)]['sequences']['flow']
wind_flow     = results[(wind, bus_electricity)]['sequences']['flow']
el_input      = results[(bus_electricity, electrolyser)]['sequences']['flow']
el_output     = results[(electrolyser, bus_H2)]['sequences']['flow']
h2_demand_ts  = results[(bus_H2, H2Demand)]['sequences']['flow']
curtailment   = results[(bus_electricity, electricity_curtailement)]['sequences']['flow']
battery_soc   = results[(battery, None)]['sequences']['storage_content']
bat_charge    = results[(bus_electricity, battery)]['sequences']['flow'] 
bat_discharge = results[(battery, bus_electricity)]['sequences']['flow']
saltcavern_soc   = results[(saltcavern, None)]['sequences']['storage_content']
saltcavern_charge    = results[(bus_H2, saltcavern)]['sequences']['flow']
saltcavern_discharge = results[(saltcavern, bus_H2)]['sequences']['flow']

# Drop the last row (oemof adds an extra timestep due to infer_last_interval)
idx = pv_flow.index[:-1]
pv_flow       = pv_flow.iloc[:-1]
wind_flow     = wind_flow.iloc[:-1]
el_input      = el_input.iloc[:-1]
el_output     = el_output.iloc[:-1]
h2_demand_ts  = h2_demand_ts.iloc[:-1]
curtailment   = curtailment.iloc[:-1]
battery_soc   = battery_soc.iloc[:-1]
bat_charge    = bat_charge.iloc[:-1]
bat_discharge = bat_discharge.iloc[:-1]
saltcavern_soc   = saltcavern_soc.iloc[:-1]
saltcavern_charge    = saltcavern_charge.iloc[:-1]
saltcavern_discharge = saltcavern_discharge.iloc[:-1]


# =============================================================================
# PLOT 1 — Annual energy flows (monthly aggregated for readability)
# =============================================================================

monthly = pd.DataFrame({
    "PV":          pv_flow.values,
    "Wind":        wind_flow.values,
    "Electrolyser":el_input.values,
    "Curtailment": curtailment.values,
    "H2 Demand":   h2_demand_ts.values,
}, index=idx).resample("ME").sum()

'''fig, ax = plt.subplots(figsize=(14, 5))
monthly[["PV", "Wind", "Electrolyser", "Curtailment"]].plot(
    kind="bar", ax=ax, width=0.8
)
#ax.set_title("Monthly Energy Flows")
ax.set_ylabel("Energy / [MWh]")
ax.set_xlabel("")
ax.set_xticklabels([d.strftime("%b") for d in monthly.index], rotation=45)
ax.legend(loc="best")
plt.tight_layout()





# Define Figures folder
figures_path = DATA_DIR.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)  # create folder if it doesn't exist

# Define output file path
output_file = figures_path / "CAVERN_monthly_flows.pdf"

# Save figure
plt.tight_layout()
func.savefigure(fig, "CAVERN_monthly_flows")
plt.show()'''



# =============================================================================
# PLOT 2 — Sample week: hourly dispatch
# =============================================================================

'''# Pick a representative summer week (July)
week_start = "2023-07-10"
week_end   = "2023-07-17"
sl = slice("2023-01-10", "2023-01-17")

fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

# Electricity bus
axes[0].fill_between(wind_flow.loc[sl].index, wind_flow.loc[sl].values, pv_flow.loc[sl].values, label="Wind", alpha=0.7)
axes[0].fill_between(pv_flow.loc[sl].index,   pv_flow.loc[sl].values,   label="PV",   alpha=0.7)
axes[0].plot(el_input.loc[sl].index,    el_input.loc[sl].values,    label="Electrolyser input", color="black", lw=1.5)
axes[0].plot(curtailment.loc[sl].index, curtailment.loc[sl].values, label="Curtailment",        color="red",   lw=1, ls="--")
axes[0].set_ylabel("MW")
#axes[0].set_title(f"Electricity Bus — {week_start} to {week_end}")
axes[0].legend(loc="upper right")

# H2 bus
axes[1].plot(el_output.loc[sl].index,    el_output.loc[sl].values,    label="H2 produced", color="green",  lw=1.5)
axes[1].plot(h2_demand_ts.loc[sl].index, h2_demand_ts.loc[sl].values, label="H2 demand",   color="orange", lw=1.5, ls="--")
axes[1].set_ylabel("MWh/h")
#axes[1].set_title(f"H2 Bus — {week_start} to {week_end}")
axes[1].legend(loc="upper right")

# Battery SOC
axes[2].fill_between(battery_soc.loc[sl].index, battery_soc.loc[sl].values, alpha=0.5, color="purple", label="Battery SOC")
axes[2].plot(bat_charge.loc[sl].index,    bat_charge.loc[sl].values,    label="Charging",    color="blue", lw=1)
axes[2].plot(bat_discharge.loc[sl].index, bat_discharge.loc[sl].values, label="Discharging", color="red",  lw=1)
axes[2].set_ylabel("MWh")
#axes[2].set_title(f"Battery — {week_start} to {week_end}")
axes[2].legend(loc="upper right")

# Salt cavern SOC
axes[3].fill_between(saltcavern_soc.loc[sl].index, saltcavern_soc.loc[sl].values, alpha=0.5, color="brown", label="Salt Cavern SOC")
axes[3].plot(saltcavern_charge.loc[sl].index,    saltcavern_charge.loc[sl].values,    label="Injection",  color="blue", lw=1)
axes[3].plot(saltcavern_discharge.loc[sl].index, saltcavern_discharge.loc[sl].values, label="Withdrawal", color="red",  lw=1)
axes[3].set_ylabel("MWh")
#axes[3].set_title(f"Salt Cavern — {week_start} to {week_end}")
axes[3].legend(loc="upper right")

axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.tight_layout()
func.savefigure(fig, "CAVERN_sample_week_dispatch")
plt.show()
'''

# =============================================================================
# PLOT 3 — Full year electrolyser load duration curve
# =============================================================================

'''fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# Top: chronological electrolyser load across 2023
axes[0].plot(el_input.index, el_input.values, color="red", lw=0.8)
axes[0].axhline(capacity_electrolyser, color="black", ls="--", lw=1, label="Installed capacity")
#axes[0].set_title("Electrolyser Load — 2023")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Electrolyser Load / [MW]")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[0].legend()

# Bottom: load duration curve
sorted_load = el_input.sort_values(ascending=False).values
axes[1].plot(sorted_load, color="steelblue", lw=1.5)
axes[1].axhline(capacity_electrolyser, color="black", ls="--", lw=1, label="Installed capacity")
#axes[1].set_title("Electrolyser Load Duration Curve — 2023")
axes[1].set_xlabel("Hours per year (ranked)")
axes[1].set_ylabel("Electrolyser Load / [MW]")
axes[1].legend()

plt.tight_layout()
func.savefigure(fig, "CAVERN_load_duration_curve")
plt.show()
'''
# =============================================================================
# PLOT 4 — Full year Battery SOC
# =============================================================================

'''fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(battery_soc.index, battery_soc.values, alpha=0.5, color="purple")
#ax.set_title("Battery State of Charge — Full Year")
ax.set_xlabel("Date")
ax.set_ylabel("Battery SOC / [MWh]")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.tight_layout()
func.savefigure(fig, "CAVERN_battery_soc_year")
plt.show()
'''


# =============================================================================
# PLOT 5 — Full year Salt Cavern SOC
# =============================================================================

'''fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(saltcavern_soc.index, saltcavern_soc.values, alpha=0.5, color="brown")
#ax.set_title("Salt Cavern State of Charge — Full Year")
ax.set_xlabel("Date")
ax.set_ylabel("Salt Cavern SOC / [MWh]")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.tight_layout()
func.savefigure(fig, "CAVERN_saltcavern_soc_year")
plt.show()'''

# =============================================================================
# PLOT 6 — Sensitivity: PV vs Wind installed capacity, cost and LCOH2 heatmaps
# Sweeps PV and Wind CAPEX, records optimal installed capacities,
# total system cost, and LCOH2
# =============================================================================

#Still to do: add LCOH2 calculation and heatmap plotting. 



# =============================================================================
# PLOT 7 — LCOH2 CALCULATION
# =============================================================================

total_cost = model.objective() # €/yr  — OPEX + annualized CAPEX of 1 representative year

total_h2_mwh = el_output.sum() # MWh_H2/yr — produced in that same year
total_h2_kg  = total_h2_mwh / par.H2_CALORIFIC_VALUE_LHV # kg_H2/yr 

lcoh2_mwh = total_cost / total_h2_mwh # €/MWh_H2
lcoh2_kg  = total_cost / total_h2_kg  # €/kg_H2

print("=" * 52)
print("  LCOH2 RESULTS")
print("=" * 52)
print(f"  Total H2 produced  : {total_h2_mwh:>12.1f} MWh_H2/yr")
print(f"  Total H2 produced  : {total_h2_kg:>12.1f} kg_H2/yr")
print(f"  Total system cost  : {total_cost:>12.0f} €/yr")
print(f"  LCOH2              : {lcoh2_mwh:>12.2f} €/MWh_H2")
print(f"  LCOH2              : {lcoh2_kg:>12.2f} €/kg_H2")
print("=" * 52)

# ── Cost breakdown — manual cross-check ──────────────────────────────────────
ep_costs_map = {
    "pv": (
        economics.annuity(capex = par.PV_CAPEX_PANELS, n =  par.PV_LIFETIME_SYSTEM, wacc = par.WACC)
        + economics.annuity(capex = par.PV_CAPEX_INVERTER, n = par.PV_LIFETIME_SYSTEM, wacc = par.WACC, u=par.PV_LIFETIME_INVERTER)
        + par.PV_OPEX
    ),
    "wind": (
        economics.annuity(capex = par.WIND_CAPEX, n = par.WIND_LIFETIME_SYSTEM, wacc = par.WACC)
        + par.WIND_OPEX
    ),
    "electrolyser": (
        economics.annuity(capex = par.ELECTROLYSER_CAPEX_SYSTEM, n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC)
        + economics.annuity(capex = par.ELECTROLYSER_CAPEX_STACK, n=par.ELECTROLYSER_LIFETIME_SYSTEM, wacc=par.WACC, u=par.ELECTROLYSER_LIFETIME_STACK)
        + par.ELECTROLYSER_OPEX
    ),
    "battery_energy": ( #€/MWh/year
        economics.annuity(capex = par.BATTERY_CAPEX_SYSTEM, n = par.LIFETIME, wacc = par.WACC, u = par.BATTERY_LIFETIME_SYSTEM) 
        + economics.annuity(capex = par.BATTERY_CAPEX_BATTERY_PACK, n = par.LIFETIME, wacc = par.WACC, u = par.BATTERY_LIFETIME_BATTERY_PACK)
    ),
    "battery_power": ( #€/MW/year
        economics.annuity(capex = par.BATTERY_CAPEX_POWER, n = par.LIFETIME, wacc = par.WACC, u = par.BATTERY_LIFETIME_SYSTEM)    
    + par.BATTERY_OPEX_FIX
    ),
    "saltcavern_energy": ( #€/MWh/year
        economics.annuity(capex = par.SALTCAVERN_CAPEX, n = par.SALTCAVERN_LIFETIME, wacc = par.WACC)
        + economics.annuity(capex = par.SALTCAVERN_CAPEX_CUSHION_GAS, n = par.SALTCAVERN_LIFETIME, wacc = par.WACC)
        + par.SALTCAVERN_OPEX
    ),
    "saltcavern_power": (
        economics.annuity(capex =par.SALTCAVERN_CAPEX_COMPRESSOR, n = par.SALTCAVERN_LIFETIME, wacc = par.WACC, u=par.SALTCAVERN_LIFETIME_COMPRESSOR)
    ),
}

invest_sizes = {
    "pv":                   results[(pv, bus_electricity)]["scalars"]["invest"],
    "wind":                 results[(wind, bus_electricity)]["scalars"]["invest"],
    "electrolyser":         results[(bus_electricity, electrolyser)]["scalars"]["invest"],
    "battery_energy":       results[(battery, None)]["scalars"]["invest"], #MWh
    "battery_power":        results[(bus_electricity, battery)]["scalars"]["invest"],
    "saltcavern_energy":    results[(saltcavern, None)]["scalars"]["invest"], #MWh
    "saltcavern_power":     results[(bus_H2, saltcavern)]["scalars"]["invest"],
}
dt = 1 # hours per timestep

variable_costs_total = {
    "battery_charge":       (bat_charge   * dt * par.BATTERY_OPEX_VAR / 2).sum(),
    "battery_discharge":    (bat_discharge * dt * par.BATTERY_OPEX_VAR / 2).sum(),
    "saltcavern_compress":  (saltcavern_charge * dt * par.SALTCAVERN_OPEX_COMPRESSOR).sum(),
}


group_map = {
    # Fixed
    "pv": ("PV", "fixed"),
    "wind": ("Wind", "fixed"),
    "electrolyser": ("Electrolyser", "fixed"),
    "battery_energy": ("Battery", "fixed"),
    "battery_power": ("Battery", "fixed"),
    "saltcavern_energy": ("Salt Cavern", "fixed"),
    "saltcavern_power": ("Salt Cavern", "fixed"),

    # Variable
    "battery_charge": ("Battery", "variable"),
    "battery_discharge": ("Battery", "variable"),
    "saltcavern_compress": ("Salt Cavern", "variable"),
}




print("\n" + "=" * 55)
print("  COST BREAKDOWN")
print("=" * 55)
print(f"\n  {'Component':<35} {'Size':>10}  {'Cost (€/yr)':>12}")
print(f"  {'-'*57}")

total_manual = 0
grouped = defaultdict(lambda: {"fixed": 0, "variable": 0})

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



fixed_labels = []
fixed_values = []

for tech, costs in grouped.items():
    if costs["fixed"] > 0:
        fixed_labels.append(tech)
        fixed_values.append(costs["fixed"])

# Sort
fixed_labels, fixed_values = zip(*sorted(
    zip(fixed_labels, fixed_values),
    key=lambda x: x[1],
    reverse=True
))

'''fig, ax = plt.subplots(figsize=(7, 7))

ax.pie(fixed_values, labels=fixed_labels, autopct='%1.1f%%', startangle=90)
ax.set_title("Fixed Costs Breakdown")
ax.axis('equal')
func.savefigure(fig, "CAVERN_fixed_costs_pie")

plt.show()'''

total_labels = []
total_values = []

for tech, costs in grouped.items():
    total = costs["fixed"] + costs["variable"]
    if total > 0:
        total_labels.append(tech)
        total_values.append(total)

# Sort
total_labels, total_values = zip(*sorted(
    zip(total_labels, total_values),
    key=lambda x: x[1],
    reverse=True
))

'''fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(total_values, labels=total_labels, autopct='%1.1f%%', startangle=90)
ax.set_title("Total Costs Breakdown (Fixed + Variable)")
ax.axis('equal')
func.savefigure(fig, "CAVERN_total_costs_pie")
plt.show()'''


print("\n" + "=" * 80)
print("  FIXED vs VARIABLE COST SPLIT")
print("=" * 80)

for tech, costs in grouped.items():
    fixed = costs["fixed"]
    variable = costs["variable"]
    total = fixed + variable

    if total == 0:
        continue

    fixed_pct = 100 * fixed / total
    var_pct = 100 * variable / total

    print(
        f"{tech:<15} | "
        f"fixed: {fixed:>12.0f} € ({fixed_pct:>5.1f}%) ; "
        f"variable: {variable:>12.0f} € ({var_pct:>5.1f}%)"
    )


# ── Instantaneous LCOH2 ───────────────────────────────────────────────────────

# ── Fixed costs: spread uniformly across all hours ────────────────────────────
# These are capacity-based annual charges (CAPEX annuities + fixed OPEX)
# and cannot be assigned to specific hours — uniform allocation is correct
total_fixed_cost   = total_cost - sum(variable_costs_total.values())  # €/yr
print(f"Sum of variable costs: {sum(variable_costs_total.values())}")
fixed_cost_per_hour = total_fixed_cost / len(el_output)               # €/h (constant)

fixed_cost_ts = pd.Series(fixed_cost_per_hour, index=el_output.index) # €/h

# ── Variable costs: reconstructed hour by hour ────────────────────────────────
battery_var_cost_ts = (
    (bat_charge + bat_discharge) *dt * par.BATTERY_OPEX_VAR / 2
)  # €/h at each timestep

saltcavern_var_cost_ts = (
    saltcavern_charge * dt * par.SALTCAVERN_OPEX_COMPRESSOR
)  # €/h at each timestep

# ── Total cost at each timestep ───────────────────────────────────────────────
total_cost_ts = (
    fixed_cost_ts
    + battery_var_cost_ts
    + saltcavern_var_cost_ts
)  # €/h

# ── Instantaneous LCOH2 ───────────────────────────────────────────────────────
# Divide hourly cost by hourly H2 production
# Replace 0 production with NaN to avoid division by zero
h2_production_safe   = el_output.replace(0, np.nan)
lcoh2_instantaneous  = total_cost_ts / h2_production_safe  # €/MWh_H2

# ── Sanity check: annual average should equal annual LCOH2 ───────────────────
lcoh2_check = total_cost_ts.sum() / el_output.sum()
print("\n" + "=" * 55)
print(f"  LCOH2 (annual, from model.objective) : {lcoh2_mwh:.2f} €/MWh_H2")
print(f"  LCOH2 (reconstructed timeseries sum) : {lcoh2_check:.2f} €/MWh_H2")
print(f"  Difference                           : {abs(lcoh2_mwh - lcoh2_check):.6f} €/MWh_H2")
print("=" * 55)



# =============================================================================
# PLOT 8 — Instantaneous LCOH2 + H2 Demand
# =============================================================================

'''fig, axes = plt.subplots(
    2, 1,
    figsize=(6, 6),
    sharex=True,
)
fig.subplots_adjust(hspace=0.08)

# ── Top: instantaneous LCOH2
axes[0].plot(
    lcoh2_instantaneous.index, lcoh2_instantaneous.values,
    color="steelblue", linewidth=0.6, label=r"Instantaneous LCOH$_2$"
)
axes[0].axhline(
    lcoh2_mwh,
    color="firebrick", linewidth=1.0, linestyle="--",
    label=rf"Annual LCOH$_2$ = {lcoh2_mwh:.2f} €/MWh$_{{H_2}}$"
)
axes[0].set_ylabel(r"LCOH$_2$ / [€ / MWh$_{H_2}$]")
axes[0].legend(loc="upper right")
#axes[0].set_title(r"Instantaneous LCOH$_2$ and H$_2$ Demand")

# ── Bottom: H2 production (electrolyser output) 
axes[1].plot(
    el_output.index, el_output.values,
    color="darkorange", linewidth=0.6, label=r"H$_2$ production"
)
axes[1].set_ylabel(r"H$_2$ Production / [MWh$_{H_2}$ / h]")
axes[1].set_xlabel(r"Date")
axes[1].legend(loc="upper right")

# ── X-axis 
locator   = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
axes[1].xaxis.set_major_locator(locator)
axes[1].xaxis.set_major_formatter(formatter)

func.savefigure(fig, "CAVERN_lcoh2_instantaneous")
plt.show()'''

print("\n" + "=" * 55)
print(f"WACC                : {par.WACC}")
print(f"WIND_LIFETIME_SYSTEM: {par.WIND_LIFETIME_SYSTEM}")
print(f"Wind annuity        : {economics.annuity(par.WIND_CAPEX, par.WIND_LIFETIME_SYSTEM, par.WACC):,.0f} €/MW/yr")
print(f"Wind ep_costs total : {economics.annuity(par.WIND_CAPEX, par.WIND_LIFETIME_SYSTEM, par.WACC) + par.WIND_OPEX:,.0f} €/MW/yr")
print("=" * 55)

# =============================================================================
# PLOT 9 — Replacement cost overhead for key components
# =============================================================================

components = {
    "PV\nInverter": {
        "capex":    par.PV_CAPEX_INVERTER,
        "n":        par.PV_LIFETIME_SYSTEM,
        "u":        par.PV_LIFETIME_INVERTER,
        "unit":     "€/MW/yr",
    },
    "Electrolyser\nStack": {
        "capex":    par.ELECTROLYSER_CAPEX_STACK,
        "n":        par.ELECTROLYSER_LIFETIME_SYSTEM,
        "u":        par.ELECTROLYSER_LIFETIME_STACK,
        "unit":     "€/MW/yr",
    },
    "Battery\nPack": {
        "capex":    par.BATTERY_CAPEX_BATTERY_PACK,
        "n":        par.BATTERY_LIFETIME_SYSTEM,
        "u":        par.BATTERY_LIFETIME_BATTERY_PACK,
        "unit":     "€/MWh/yr",
    },
    "Salt Cavern\nCompressor": {
        "capex":    par.SALTCAVERN_CAPEX_COMPRESSOR,
        "n":        par.SALTCAVERN_LIFETIME,
        "u":        par.SALTCAVERN_LIFETIME_COMPRESSOR,
        "unit":     "€/MW/yr",
    },
}

# ── Compute annuities with and without replacement ────────────────────────────
labels, without_replacement, with_replacement, overhead, no_replacements = [], [], [], [], []

for name, c in components.items():
    a_without = economics.annuity(c["capex"], c["n"], par.WACC)            # u = n, no replacement
    a_with    = economics.annuity(c["capex"], c["n"], par.WACC, u=c["u"])  # with replacement
    labels.append(name)
    without_replacement.append(a_without)
    with_replacement.append(a_with)
    overhead.append(a_with - a_without)
    n_rep = max(0, int(c["n"] // c["u"]) - 1)
    no_replacements.append(n_rep)

# ── Plot ──────────────────────────────────────────────────────────────────────
x     = np.arange(len(labels))
width = 0.35

'''fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(wspace=0.35)

# Left: stacked bar — base annuity + replacement overhead
axes[0].bar(x, without_replacement, width, label="Without replacement", color="steelblue")
axes[0].bar(x, overhead,            width, label="Replacement overhead", color="firebrick",
            bottom=without_replacement)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel(r"Annualized cost / [€/unit/year]")
axes[0].set_title("Replacement cost overhead")
axes[0].legend()
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))'''

# Right: % overhead relative to no-replacement annuity
pct_overhead = [o / w * 100 for o, w in zip(overhead, without_replacement)]
'''axes[1].bar(x, pct_overhead, width, color="firebrick")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel(r"Replacement overhead / [\%]")
axes[1].set_title("Replacement cost as % of base annuity")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
'''
# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"  {'Component':<25} {'No replacement':>15} {'With replacement':>16} {'Overhead':>15} {'No. of Replacements':>22}")
print("=" * 100)
for i, name in enumerate(labels):
    print(
        f"  {name.replace(chr(10), ' '):<25}"
        f"  {without_replacement[i]:>13,.0f}"
        f"  {with_replacement[i]:>14,.0f}"
        f"  {overhead[i]:>8,.0f}  ({pct_overhead[i]:>6.1f}%)"
        f"  {no_replacements[i]:>15}"
    )
print("=" * 100)

'''func.savefigure(fig, "CAVERN_replacement_cost_overhead")
plt.show()'''


# =============================================================================
# FINANCIAL METRICS — NPV, IRR, Payback, H2 Selling Price
# =============================================================================

# ── H2 selling price ─────────────────────────────────────────────────────────
MARGIN           = par.MARGIN_H2
H2_SELLING_PRICE = lcoh2_mwh / (1 - MARGIN)               # €/MWh_H2

# ── Annual revenue ────────────────────────────────────────────────────────────
annual_revenue   = total_h2_mwh * H2_SELLING_PRICE         # €/yr

# ── Annual variable OPEX (already computed above) ────────────────────────────
annual_var_opex  = sum(variable_costs_total.values())       # €/yr

# ── Annual fixed OPEX (capacity-based, not in annuity for financial model) ───
annual_fixed_opex = (
      invest_sizes["pv"]                      * par.PV_OPEX
    + invest_sizes["wind"]                    * par.WIND_OPEX
    + invest_sizes["electrolyser"]            * par.ELECTROLYSER_OPEX
    + invest_sizes["battery_power"]           * par.BATTERY_OPEX_FIX
    + invest_sizes["saltcavern_energy"]       * par.SALTCAVERN_OPEX
)

annual_cost_total = annual_fixed_opex + annual_var_opex     # €/yr
annual_cashflow   = annual_revenue - annual_cost_total      # €/yr net of OPEX

# ── Total upfront CAPEX (undiscounted, as spent at year 0) ───────────────────
capex_total = (
      invest_sizes["pv"]                 * (par.PV_CAPEX_PANELS + par.PV_CAPEX_INVERTER)
    + invest_sizes["wind"]               * par.WIND_CAPEX
    + invest_sizes["electrolyser"]       * par.ELECTROLYSER_CAPEX
    + invest_sizes["battery_energy"]     * par.BATTERY_CAPEX
    + invest_sizes["battery_power"]      * par.BATTERY_CAPEX_POWER
    + invest_sizes["saltcavern_energy"]  * (par.SALTCAVERN_CAPEX + par.SALTCAVERN_CAPEX_CUSHION_GAS)
    + invest_sizes["saltcavern_power"]   * par.SALTCAVERN_CAPEX_COMPRESSOR
)
print("Total CAPEX (undiscounted):", capex_total, "€")
print("Total CAPEX (discounted, annuity):", total_fixed_cost, "€ per year")
# ── Project horizon ───────────────────────────────────────────────────────────
horizon = par.LIFETIME

# ── Cash flow array ───────────────────────────────────────────────────────────
# Year 0: -CAPEX; years 1..horizon: net annual cashflow
#1st value is UNDISCOUNTED CAPEX because NPV refers to the value of the investment in the present (year 0), so we use the actual upfront CAPEX cost.
cashflows = np.array([-capex_total] + [annual_cashflow] * horizon) #creates array with -capex_total followed by annual_cashflow repeated 'horizon' times
print("\nCashflows (year 0 to horizon):", cashflows)

# ── NPV time series ───────────────────────────────────────────────────────────
years      = np.arange(0, horizon + 1)
print("Years:", years)
npv_series = np.array([
    npf.npv(par.WACC, cashflows[:t + 1])
    for t in range(len(cashflows))
])
print("NPV series (year 0 to horizon):", npv_series)

# ── Scalar metrics ────────────────────────────────────────────────────────────
npv_final  = npf.npv(par.WACC, cashflows)
irr        = npf.irr(cashflows)

# ── Discounted payback: linear interpolation between last negative and first  # REMOVE THIS BLOCK to show only simple PBT
#    positive NPV value                                                         # REMOVE
payback_idx_disc = np.where(npv_series >= 0)[0]                                # REMOVE
if len(payback_idx_disc) > 0:                                                  # REMOVE
    t0_d = payback_idx_disc[0] - 1                                             # REMOVE
    t1_d = payback_idx_disc[0]                                                 # REMOVE
    frac_disc        = -npv_series[t0_d] / (npv_series[t1_d] - npv_series[t0_d])  # REMOVE
    payback_discounted = t0_d + frac_disc                                      # REMOVE
else:                                                                           # REMOVE
    payback_discounted = None                                                   # REMOVE

print("\n" + "=" * 52)
print("  FINANCIAL METRICS")
print("=" * 52)
print(f"  LCOH2              : {lcoh2_mwh:>10.2f} €/MWh_H2")
print(f"  Margin             : {MARGIN*100:>10.1f} %")
print(f"  H2 selling price   : {H2_SELLING_PRICE:>10.2f} €/MWh_H2")
print(f"  Profit per MWh     : {H2_SELLING_PRICE - lcoh2_mwh:>10.2f} €/MWh_H2")
print(f"  Annual revenue     : {annual_revenue:>12.0f} €/yr")
print(f"  Annual OPEX total  : {annual_cost_total:>12.0f} €/yr")
print(f"  Annual net cashflow: {annual_cashflow:>12.0f} €/yr")
print(f"  Total CAPEX        : {capex_total:>12.0f} €")
print(f"  Project horizon    : {horizon:>10} years")
print(f"  NPV ({horizon} yr)      : {npv_final/1e6:>10.2f} M€")
print(f"  IRR                : {irr*100:>10.2f} %")
if payback_discounted is not None:                                               
    print(f"  Payback time (discounted): {payback_discounted:>9.2f} years  (NPV zero-crossing)")  
else:                                                                            
    print(f"  Payback time (discounted):   >{horizon} years (not recovered)")        
print("=" * 52)

# =============================================================================
# PLOT 10 — NPV evolution over project lifetime
# =============================================================================

'''fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(years, npv_series / 1e6, color="steelblue", linewidth=1.0, label="NPV")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

# Discounted payback — interpolated zero-crossing, marker at y=0              
if payback_discounted is not None:
    ax.axvline(
        payback_discounted, color="black", linewidth=0.8, linestyle="--",  
        label=f"Payback Time (discounted) = {payback_discounted:.1f} yrs"             
    )                                                              
    ax.plot(payback_discounted, 0, "o", color="black", markersize=5)

ax.set_xlabel(r"Year")
ax.set_ylabel(r"NPV / [M€]")
ax.set_title(r"Net Present Value over Project Lifetime")
ax.legend()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

func.savefigure(fig, "CAVERN_npv_evolution")
plt.show()'''




# =============================================================================
# LOGBOOK — CSV
# =============================================================================

logbook = {

    "scenario": SCENARIO_NAME,

    # ── Key model parameters ──────────────────────────────────────────────────
    "parameters": {
        "wacc":                              par.WACC,
        "lifetime_yr":                       par.LIFETIME,
        "margin_h2":                         par.MARGIN_H2,
        "pv_lifetime_system_yr":             par.PV_LIFETIME_SYSTEM,
        "pv_lifetime_inverter_yr":           par.PV_LIFETIME_INVERTER,
        "wind_lifetime_yr":                  par.WIND_LIFETIME_SYSTEM,
        "electrolyser_efficiency":           par.ELECTROLYSER_EFFICIENCY,
        "electrolyser_min_load":             par.ELECTROLYSER_MIN_LOAD,
        "electrolyser_lifetime_system_yr":   par.ELECTROLYSER_LIFETIME_SYSTEM,
        "electrolyser_lifetime_stack_yr":    par.ELECTROLYSER_LIFETIME_STACK,
        "battery_efficiency":                par.BATTERY_EFFICIENCY_RT,
        "battery_lifetime_system_yr":        par.BATTERY_LIFETIME_SYSTEM,
        "battery_lifetime_pack_yr":          par.BATTERY_LIFETIME_BATTERY_PACK,
        "saltcavern_lifetime_yr":            par.SALTCAVERN_LIFETIME,
        "saltcavern_lifetime_compressor_yr": par.SALTCAVERN_LIFETIME_COMPRESSOR,
        # ← add new unit parameters here
    },

    # ── Optimal capacities ────────────────────────────────────────────────────
    "capacities": {
        "capacity_pv_mw":                    float(capacity_pv),
        "capacity_wind_mw":                  float(capacity_wind),
        "capacity_total_renewable_mw":       float(capacity_pv + capacity_wind),
        "capacity_electrolyser_mw":          float(capacity_electrolyser),
        "capacity_battery_energy_mwh":       float(capacity_battery),
        "capacity_battery_power_mw":         float(invest_sizes["battery_power"]),
        "capacity_saltcavern_energy_mwh":    float(capacity_saltcavern),
        "capacity_saltcavern_compressor_mw": float(invest_sizes["saltcavern_power"]),
        # ← add hp_mw, chp_mw, heat_storage_mwh here
    },

    # ── Operational metrics ───────────────────────────────────────────────────
    "operational": {
        "capacity_factor_pv":               func.capacity_factor(pv_flow, capacity_pv, dt)   if capacity_pv   > 0 else 0.0,
        "capacity_factor_wind":             func.capacity_factor(wind_flow, capacity_wind, dt) if capacity_wind > 0 else 0.0,
        "capacity_factor_res_avg":          func.capacity_factor(pv_flow + wind_flow, capacity_pv + capacity_wind, dt) if (capacity_pv + capacity_wind) > 0 else 0.0,
        "capacity_factor_electrolyser":     func.capacity_factor(el_input, capacity_electrolyser, dt) if capacity_electrolyser > 0 else 0.0,
        "total_h2_produced_mwh":            float(total_h2_mwh),
        "total_h2_produced_kg":             float(total_h2_kg),
        "total_renewable_gen_mwh":          float(pv_flow.sum() + wind_flow.sum()),
        "curtailment_mwh":                  float(curtailment.sum()),
        "curtailment_fraction":             float(curtailment.sum() / (pv_flow.sum() + wind_flow.sum())) if (pv_flow.sum() + wind_flow.sum()) > 0 else 0.0,
        "capacity_factor_battery_energy":   func.capacity_factor(battery_soc, invest_sizes["battery_energy"], dt) if invest_sizes["battery_energy"] > 0 else 0.0,
        "capacity_factor_battery_power":    func.capacity_factor(bat_discharge, invest_sizes["battery_power"], dt) if invest_sizes["battery_power"] > 0 else 0.0,
        "battery_charge_mw":                float(bat_charge.sum()),
        "battery_discharge_mw":             float(bat_discharge.sum()),
        "capacity_factor_saltcavern_energy":func.capacity_factor(saltcavern_soc, invest_sizes["saltcavern_energy"], dt) if invest_sizes["saltcavern_energy"] > 0 else 0.0,
        "capacity_factor_saltcavern_power": func.capacity_factor(saltcavern_discharge, invest_sizes["saltcavern_power"], dt) if invest_sizes["saltcavern_power"] > 0 else 0.0,
        "saltcavern_charge_mw":             float(saltcavern_charge.sum()),
        "saltcavern_discharge_mw":          float(saltcavern_discharge.sum()),
        # ← add heat flows here
    },

    # ── Levelized Costs ───────────────────────────────────────────────────────
    "levelized_cost": {
        "lcoh2_eur_per_mwh":            float(lcoh2_mwh),
        "lcoh2_eur_per_kg":             float(lcoh2_kg),
    },

    # ── Commercial ────────────────────────────────────────────────────────────
    "commercial": {
        "margin_applied":           float(MARGIN),
        "h2_selling_price_eur_per_mwh": float(H2_SELLING_PRICE),
        "profit_eur_per_mwh":       float(H2_SELLING_PRICE - lcoh2_mwh),
        "annual_revenue_eur":       float(annual_revenue),
    },

    # ── CAPEX undiscounted ────────────────────────────────────────────────────
    "capex_undiscounted": {
        "pv_panels_eur":             float(invest_sizes["pv"]            * par.PV_CAPEX_PANELS),
        "pv_inverter_eur":           float(invest_sizes["pv"]            * par.PV_CAPEX_INVERTER),
        "pv_total_eur":              float(invest_sizes["pv"]            * (par.PV_CAPEX_PANELS + par.PV_CAPEX_INVERTER)),
        "wind_eur":                  float(invest_sizes["wind"]          * par.WIND_CAPEX),
        "electrolyser_system_eur":   float(invest_sizes["electrolyser"]  * par.ELECTROLYSER_CAPEX_SYSTEM),
        "electrolyser_stack_eur":    float(invest_sizes["electrolyser"]  * par.ELECTROLYSER_CAPEX_STACK),
        "electrolyser_total_eur":    float(invest_sizes["electrolyser"]  * par.ELECTROLYSER_CAPEX),
        "battery_system_eur":        float(invest_sizes["battery_energy"]* par.BATTERY_CAPEX_SYSTEM),
        "battery_pack_eur":          float(invest_sizes["battery_energy"]* par.BATTERY_CAPEX_BATTERY_PACK),
        "battery_total_eur":         float(invest_sizes["battery_energy"]* par.BATTERY_CAPEX),
        "saltcavern_geological_eur": float(invest_sizes["saltcavern_energy"] * par.SALTCAVERN_CAPEX),
        "saltcavern_cushiongas_eur": float(invest_sizes["saltcavern_energy"] * par.SALTCAVERN_CAPEX_CUSHION_GAS),
        "saltcavern_compressor_eur": float(invest_sizes["saltcavern_power"]  * par.SALTCAVERN_CAPEX_COMPRESSOR),
        "total_eur":                 float(capex_total),
        # ← add new unit CAPEX here
    },

    # ── Annualized CAPEX (ep_costs × size, €/yr) ─────────────────────────────
    "capex_annualized_eur_per_yr": {
        k: float(ep_costs_map[k] * invest_sizes[k]) for k in ep_costs_map
    },

    # ── Fixed OPEX (€/yr) ─────────────────────────────────────────────────────
    "opex_fixed_eur_per_yr": {
        "pv":           float(invest_sizes["pv"]            * par.PV_OPEX),
        "wind":         float(invest_sizes["wind"]          * par.WIND_OPEX),
        "electrolyser": float(invest_sizes["electrolyser"]  * par.ELECTROLYSER_OPEX),
        "battery":      float(invest_sizes["battery_power"] * par.BATTERY_OPEX_FIX),
        "saltcavern":   float(invest_sizes["saltcavern_energy"] * par.SALTCAVERN_OPEX),
        "total":        float(annual_fixed_opex),
        # ← add new unit fixed OPEX here
    },

    # ── Variable OPEX (€/yr) ──────────────────────────────────────────────────
    "opex_variable_eur_per_yr": {
        "battery_charge":      float(variable_costs_total["battery_charge"]),
        "battery_discharge":   float(variable_costs_total["battery_discharge"]),
        "saltcavern_compress": float(variable_costs_total["saltcavern_compress"]),
        "total":               float(annual_var_opex),
        # ← add new variable costs here
    },

    # ── Annual cashflow model ─────────────────────────────────────────────────
    "cashflow": {
        "annual_fixed_opex_eur":        float(annual_fixed_opex),
        "annual_variable_opex_eur":     float(annual_var_opex),
        "annual_total_opex_eur":        float(annual_cost_total),
        "annual_revenue_eur":           float(annual_revenue),
        "annual_net_cashflow_eur":      float(annual_cashflow), #annual_revenue - annual_cost_total (€/yr net of OPEX)
        "total_capex_undiscounted_eur": float(capex_total),
    },

    # ── Financial metrics ─────────────────────────────────────────────────────
    "financial": {
        "total_system_cost_eur_per_yr": float(total_cost), # of 1 year (CAPEX annuity + OPEX)
        "npv_final_eur":         float(npv_final),
        "irr":                   float(irr)                if irr                is not None else None,
        "payback_discounted_yr": float(payback_discounted) if payback_discounted is not None else None,
        "project_horizon_yr":    int(horizon),
    },
}

func.save_logbook_csv(logbook, DATA_DIR, SCENARIO_NAME)
