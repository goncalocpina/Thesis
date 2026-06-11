import math as m
from CoolProp.CoolProp import PropsSI


# =============================================================================
# Supply and demand operational parameters
# =============================================================================
PV_CAPACITY = 40 # MW
WIND_CAPACITY = 200 # MW

# =============================================================================
# CURTAILMENT PARAMETERS
# =============================================================================

CURTAILMENT_SHARE = 0.0 # N% curtailment of renewable generation (PV + Wind) as a starting point for flexibility needs assessment

# =============================================================================
# Techno-economic PARAMETERS
# =============================================================================
USD_to_EUR =  0.85 #1 USD = 0.85 EUR
MARGIN = 8.2/100 # 8.2% margin for profit (SALES PRICE = LCOX / (1 - MARGIN))
LIFETIME = 20
WACC = 0.083616 # Weighted Average Cost of Capital (WACC) for annuity calculations (7% is a common assumption for renewable energy projects)

#WACC = E*Re + D*Rd*(1-Tc)
# where:
# E = 0.4
# D = 0.6
# Re = 0.12 (cost of equity)
# Rd = 0.08 (cost of debt)
#https://www.lazard.com/media/5tlbhyla/lazards-lcoeplus-june-2025-_vf.pdf
# search for wacc, and also look into pdf pg 42
# Tc = 0.258 (corporate tax rate)
#https://taxsummaries.pwc.com/netherlands/corporate/taxes-on-corporate-income 

# =============================================================================
# PV
# =============================================================================
PV_CAPEX = 628000                               # €/MW (total system cost, including panels, inverter, and EPC)
PV_CAPEX_INVERTER = 103000                      # €/MW (inverter-only cost, excluding PV panels and EPC)
PV_CAPEX_PANELS = PV_CAPEX - PV_CAPEX_INVERTER  # €/MW, self-explanatory
PV_OPEX = PV_CAPEX*1.3/100                      # €/MW/year 
PV_LIFETIME_SYSTEM = 20                         # years
PV_LIFETIME_INVERTER = 15                       # years (inverters typically need replacement after 15 years)

# =============================================================================
# WIND
# =============================================================================
WIND_CAPEX = 1325000          # €/MW
WIND_OPEX = 45000             # €/MW/year
WIND_LIFETIME_SYSTEM = 20     # years

# =============================================================================
# ELECTROLYSER
# =============================================================================
ELECTROLYSER_EFFICIENCY = 0.65        # 65% efficiency (LHV basis)
ELECTROLYSER_MIN_LOAD = 0.2           # Minimum load as a fraction of capacity (20%)
ELECTROLYSER_RECOVERABLE_HEAT = 0.3  # Fraction of the total input energy recoverable as heat (25%)
ELECTROLYSER_CAPEX = 1666000          # €/MW    
ELECTROLYSER_CAPEX_STACK = 408000     # €/MW (stack-only cost, excluding BoP and EPC)
ELECTROLYSER_CAPEX_SYSTEM = ELECTROLYSER_CAPEX - ELECTROLYSER_CAPEX_STACK
ELECTROLYSER_OPEX = 43000             # €/MW/year 
ELECTROLYSER_LIFETIME_SYSTEM = 20     # years
ELECTROLYSER_LIFETIME_STACK = 5       # years (stacks typically need replacement after 5 years)

# =============================================================================
# HEAT PUMP
# =============================================================================
HEATPUMP_COP = 3.7                  # Coefficient of Performance (COP) for the heat pump
HEATPUMP_COP_SUMMERFACTOR = 1       # Summer performance factor (heat pump operates at 100% of its rated COP during summer)
HEATPUMP_SUMMERTIME_START = 5       # Month when summer performance starts (May)
HEATPUMP_SUMMERTIME_END = 10        # Month when summer performance ends (October)
HEATPUMP_COP_WINTERFACTOR = 0.7     # Winter performance factor (heat pump operates at 70% of its rated COP during winter)
HEATPUMP_MIN_LOAD = 0.15            # Minimum load as a fraction of capacity (15%)
HEATPUMP_CAPEX = 600000             # €/MW
HEATPUMP_OPEX_FIX = 2700            # €/MW/year (fixed OPEX for power capacity)
HEATPUMP_OPEX_VAR = 1.7             # €/MWh (variable OPEX based on energy throughput)
HEATPUMP_LIFETIME = 20              # years

# =============================================================================
# FUEL CELL
# =============================================================================
FUELCELL_EFFICIENCY_HEAT = 0.4          # 40% efficiency (LHV basis)
FUELCELL_EFFICIENCY_ELECTRICITY = 0.5   # 50% efficiency (LHV basis)
FUELCELL_MIN_LOAD = 0.3                 # Minimum load as a fraction of capacity
FUELCELL_CAPEX = 3500000                # €/MW
FUELCELL_CAPEX_STACK = 875000           # €/MW (stack-only cost, excluding BoP and EPC)
FUELCELL_CAPEX_SYSTEM = FUELCELL_CAPEX - FUELCELL_CAPEX_STACK  # €/MW (system cost excluding stack)
FUELCELL_OPEX_VAR = 50                  # €/MWh (stack-only cost, excluding BoP and EPC)
FUELCELL_OPEX_FIX = 0.05 * FUELCELL_CAPEX  # €/MW/year (fixed OPEX)
FUELCELL_LIFETIME_STACK = 5             # years (stacks typically need replacement after 5 years)
FUELCELL_LIFETIME_SYSTEM = 20           # years

# =============================================================================
# BATTERY
# =============================================================================
BATTERY_EFFICIENCY_RT = 0.93                                        # 93% efficiency (round-trip, including both charging and discharging losses)
BATTERY_EFFICIENCY_CHARGE = m.sqrt(0.93)                            # Battery charging efficiency
BATTERY_EFFICIENCY_DISCHARGE = m.sqrt(0.93)                         # Battery discharging efficiency
BATTERY_SELF_DISCHARGE_RATE = 0.0028                                # %/hour, equivalent to 2%/month
BATTERY_RECOVERABLE_HEAT_RT = 0.035                                 # Fraction of the total input energy recoverable as heat (25%) in 1 RT cycle
BATTERY_MIN_CHARGE_TIME = 4.0                                       # hours for a full charge (C-rate = 1/4 = 0.25)
BATTERY_MIN_DISCHARGE_TIME = BATTERY_MIN_CHARGE_TIME                # hours for a full discharge (C-rate = 1/4 = 0.25)
BATTERY_CAPEX = 369000 * USD_to_EUR                                 # €/MWh (storage capacity)
BATTERY_CAPEX_BATTERY_PACK = 50000 * USD_to_EUR                     # €/MWh (battery pack cost, excluding BoP and EPC)
BATTERY_CAPEX_SYSTEM = BATTERY_CAPEX - BATTERY_CAPEX_BATTERY_PACK   # €/MWh (system cost excluding battery pack)
BATTERY_CAPEX_POWER = 00                                            # €/MW (charge/discharge power)    
BATTERY_OPEX_FIX = 8000 * USD_to_EUR                                # €/MW/year (fixed OPEX for power capacity)
BATTERY_OPEX_VAR = 0.3                                              # €/MWh throughput (variable OPEX based on energy throughput)
BATTERY_LIFETIME_SYSTEM = 10                                        # years    
BATTERY_LIFETIME_BATTERY_PACK = 5                                   # years (battery packs typically last around 5 years)

# =============================================================================
# SALT CAVERN
# =============================================================================
# [T] = K; [p] = Pa; [rho] = kg/m³
H2_DENSITY_IN_SALTCAVERN  = PropsSI('D', 'T', 80 + 273.15, 'P', 10e6, 'Hydrogen')
H2_DENSITY_OUT_SALTCAVERN = PropsSI('D', 'T', 25 + 273.15, 'P', 1e6 , 'Hydrogen')
H2_CALORIFIC_VALUE_LHV = 0.0333 # MWh/kg (lower heating value of H2; LHV_H2 = 33.3 kWh/kg = 0.0333 MWh/kg)

SALTCAVERN_EFFICIENCY = 1                                               # 100% efficiency (round-trip, only gas content)
SALTCAVERN_CAPEX = 21/H2_CALORIFIC_VALUE_LHV                            # €/MWh (capital cost for salt cavern storage - 2.5€/kg H2, converted to €/MWh using the calorific value)
SALTCAVERN_CAPEX_CUSHION_GAS = 2.5/H2_CALORIFIC_VALUE_LHV               # €/MWh (Cost of cushion gas - 2.5€/kg H2, converted to €/MWh using the calorific value)
SALTCAVERN_CAPEX_COMPRESSOR = 2481000 * USD_to_EUR                      # €/MW (capital cost for compression)
SALTCAVERN_OPEX = 0.04 * SALTCAVERN_CAPEX                               # €/MWh/year (OPEX for salt cavern storage is 4% of CAPEX per year; only CAPEX of the cavern itself; i.e.; geological formation site)
SALTCAVERN_OPEX_COMPRESSOR = 0.014 * USD_to_EUR/H2_CALORIFIC_VALUE_LHV  # €/MWh (OPEX for compression, 0.014 $/kg H2, converted to €/MWh using the calorific value and exchange rate)
SALTCAVERN_LIFETIME = 20                                                # years (salt caverns typically have a long lifespan)
SALTCAVERN_LIFETIME_COMPRESSOR = 15                                     # years (compressors typically have a shorter lifespan than the cavern itself)
SALTCAVERN_SELF_DISCHARGE_RATE = 0                                      # %/day, assuming it's air-tight and has negligible self-discharge over the timescales considered
SALTCAVERN_RELATIVE_VOLUME_LOSS = 0.01                                  # 1% storage volume loss per year
SALTCAVERN_CUSHIONGAS_FRACTION = 0.3                                    # Minimum SOC the cavern should have at a given moment (to maintain pressure and cavern integrity)

# =============================================================================
# THERMAL STORAGE
# =============================================================================
THERMALSTORAGE_EFFICIENCY_RT = 0.9                  # 90% efficiency (round-trip, including both charging and discharging losses)   
THERMALSTORAGE_EFFICIENCY_CHARGE = m.sqrt(THERMALSTORAGE_EFFICIENCY_RT)      # Thermal storage charging efficiency
THERMALSTORAGE_EFFICIENCY_DISCHARGE = m.sqrt(THERMALSTORAGE_EFFICIENCY_RT)
THERMALSTORAGE_SELF_DISCHARGE_RATE = 0.0017         # %/hour, equivalent to 15%/year
THERMALSTORAGE_CAPEX  = 1760000                     # €/MWh
THERMALSTORAGE_OPEX = 1700                          # €/MWh/year
THERMALSTORAGE_LIFETIME = 20                        # years
THERMALSTORAGE_MIN_CHARGE_TIME = 24                 # hours → C-rate = 0.1