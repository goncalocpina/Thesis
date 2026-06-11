"""
parameters.py
=============
Technology and economic parameters for the H2 energy system optimization model.

All monetary values are in Euros (€).
Energy values are in MWh (megawatt-hours).
Power values are in MW (megawatts).
Mass values are in kg.

Physical constant: 1 MWh_H2 = 33.3 kgH2 (lower heating value).
"""

# ---------------------------------------------------------------------------
# Discount rate and time horizon
# ---------------------------------------------------------------------------
DISCOUNT_RATE = 0.08  # [-] Weighted average cost of capital (WACC)


def crf(interest_rate: float, lifetime_years: int) -> float:
    """
    Capital Recovery Factor (CRF).

    Converts a one-time capital expenditure into an equivalent uniform
    annual cost over the asset lifetime.

    Formula:
        CRF(i, n) = i * (1+i)^n / [(1+i)^n - 1]

    Parameters
    ----------
    interest_rate : float
        Annual discount rate (e.g. 0.08 for 8%).
    lifetime_years : int
        Economic lifetime of the asset in years.

    Returns
    -------
    float
        Dimensionless annuity factor.
    """
    i = interest_rate
    n = lifetime_years
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


# ---------------------------------------------------------------------------
# PV (Photovoltaic) parameters
# ---------------------------------------------------------------------------
PV_CAPEX = 600_000          # €/MW installed capacity
PV_OPEX = 10_000            # €/MW/year fixed O&M
PV_LIFETIME = 25            # years
PV_CRF = crf(DISCOUNT_RATE, PV_LIFETIME)

# ---------------------------------------------------------------------------
# Wind turbine parameters
# ---------------------------------------------------------------------------
WIND_CAPEX = 1_200_000      # €/MW installed capacity
WIND_OPEX = 25_000          # €/MW/year fixed O&M
WIND_LIFETIME = 20          # years
WIND_CRF = crf(DISCOUNT_RATE, WIND_LIFETIME)

# ---------------------------------------------------------------------------
# Electrolyser (PEM or Alkaline) parameters
# ---------------------------------------------------------------------------
ELECTROLYSER_EFFICIENCY = 0.70   # MWh_H2 / MWh_electricity (LHV basis)
ELECTROLYSER_MIN_LOAD = 0.10     # minimum fraction of rated power [0-1]
ELECTROLYSER_CAPEX = 700_000     # €/MW electrical input capacity
ELECTROLYSER_OPEX = 21_000       # €/MW/year fixed O&M
ELECTROLYSER_LIFETIME = 15       # years
ELECTROLYSER_CRF = crf(DISCOUNT_RATE, ELECTROLYSER_LIFETIME)

# ---------------------------------------------------------------------------
# Battery storage (Li-ion) parameters
# ---------------------------------------------------------------------------
BATTERY_RT_EFFICIENCY = 0.90     # round-trip efficiency [-]
BATTERY_ETA_IN = 0.95            # charge efficiency  (sqrt of RT) [-]
BATTERY_ETA_OUT = 0.95           # discharge efficiency (sqrt of RT) [-]
BATTERY_SELF_DISCHARGE = 0.001   # self-discharge per hour [fraction/hour]
BATTERY_C_RATIO = 4.0            # energy/power ratio [hours]; C-rate = 1/C_ratio
BATTERY_CAPEX_ENERGY = 150_000   # €/MWh energy capacity
BATTERY_OPEX_POWER = 5_000       # €/MW/year fixed O&M (power side)
BATTERY_OPEX_VARIABLE = 1.0      # €/MWh cycled (variable O&M)
BATTERY_LIFETIME = 15            # years
BATTERY_CRF = crf(DISCOUNT_RATE, BATTERY_LIFETIME)

# ---------------------------------------------------------------------------
# Salt cavern H2 storage parameters
# ---------------------------------------------------------------------------
# Physical constant
H2_LHV_KWH_PER_KG = 33.3        # kWh/kgH2 (lower heating value)
H2_LHV_MWH_PER_KG = H2_LHV_KWH_PER_KG / 1000.0   # MWh/kgH2 = 0.0333

# Cavern CAPEX: geology + cushion gas
# Cushion gas = 30% of total cavity volume; it cannot be recovered.
CAVERN_CAPEX_GEOLOGY = 21.0      # €/kgH2 usable capacity
CAVERN_CAPEX_CUSHION_GAS = 2.5   # €/kgH2 usable capacity (cost of cushion gas)
CAVERN_CUSHION_GAS_FRACTION = 0.30  # fraction of total cavern volume

# Total CAPEX per kgH2 (usable working-gas capacity):
#   Geology cost + cushion-gas volume that must be purchased
#   (Cushion gas fraction applied to the cushion_gas unit cost)
CAVERN_CAPEX_PER_KG_H2 = (
    CAVERN_CAPEX_GEOLOGY
    + CAVERN_CAPEX_CUSHION_GAS * (1 + CAVERN_CUSHION_GAS_FRACTION)
)

# Convert to €/MWh_H2:  CAPEX [€/kgH2] / (H2 LHV [MWh/kgH2])
#   = CAPEX [€/kgH2] * (kgH2/MWh) = CAPEX [€/kgH2] * (1 / 0.0333 kgH2/kWh * 1000)
#   Numerically: (21 + 2.5*1.30) €/kgH2 * 33.3 kgH2/MWh
CAVERN_CAPEX_PER_MWH_H2 = CAVERN_CAPEX_PER_KG_H2 / H2_LHV_MWH_PER_KG

CAVERN_RT_EFFICIENCY = 1.0       # [-] lossless compression cycle assumed
CAVERN_VOLUME_LOSS_ANNUAL = 0.01 # 1% annual volume/inventory loss (cavern shrinkage)
CAVERN_VOLUME_LOSS_HOURLY = CAVERN_VOLUME_LOSS_ANNUAL / 8760.0  # per-hour decay rate

# OPEX = 4% of geology CAPEX per year, expressed per MWh of working-gas capacity
CAVERN_OPEX_FRACTION = 0.04      # fraction of geology CAPEX
CAVERN_OPEX_PER_MWH_H2 = (
    CAVERN_CAPEX_GEOLOGY / H2_LHV_MWH_PER_KG * CAVERN_OPEX_FRACTION
)

CAVERN_LIFETIME = 30             # years
CAVERN_CRF = crf(DISCOUNT_RATE, CAVERN_LIFETIME)

# ---------------------------------------------------------------------------
# Simulation horizon
# ---------------------------------------------------------------------------
N_HOURS = 8760  # one full year (non-leap)

# ---------------------------------------------------------------------------
# Convenience summary (used by results_extractor and sensitivity_analysis)
# ---------------------------------------------------------------------------
PARAMS = {
    "discount_rate": DISCOUNT_RATE,
    # PV
    "pv_capex": PV_CAPEX,
    "pv_opex": PV_OPEX,
    "pv_lifetime": PV_LIFETIME,
    "pv_crf": PV_CRF,
    # Wind
    "wind_capex": WIND_CAPEX,
    "wind_opex": WIND_OPEX,
    "wind_lifetime": WIND_LIFETIME,
    "wind_crf": WIND_CRF,
    # Electrolyser
    "elec_efficiency": ELECTROLYSER_EFFICIENCY,
    "elec_min_load": ELECTROLYSER_MIN_LOAD,
    "elec_capex": ELECTROLYSER_CAPEX,
    "elec_opex": ELECTROLYSER_OPEX,
    "elec_lifetime": ELECTROLYSER_LIFETIME,
    "elec_crf": ELECTROLYSER_CRF,
    # Battery
    "batt_rt_efficiency": BATTERY_RT_EFFICIENCY,
    "batt_eta_in": BATTERY_ETA_IN,
    "batt_eta_out": BATTERY_ETA_OUT,
    "batt_self_discharge": BATTERY_SELF_DISCHARGE,
    "batt_c_ratio": BATTERY_C_RATIO,
    "batt_capex_energy": BATTERY_CAPEX_ENERGY,
    "batt_opex_power": BATTERY_OPEX_POWER,
    "batt_opex_variable": BATTERY_OPEX_VARIABLE,
    "batt_lifetime": BATTERY_LIFETIME,
    "batt_crf": BATTERY_CRF,
    # Salt cavern
    "cavern_capex_per_mwh": CAVERN_CAPEX_PER_MWH_H2,
    "cavern_opex_per_mwh": CAVERN_OPEX_PER_MWH_H2,
    "cavern_rt_efficiency": CAVERN_RT_EFFICIENCY,
    "cavern_volume_loss_hourly": CAVERN_VOLUME_LOSS_HOURLY,
    "cavern_lifetime": CAVERN_LIFETIME,
    "cavern_crf": CAVERN_CRF,
    # Simulation
    "n_hours": N_HOURS,
    # Big-M for electrolyser min-load linearisation
    "elec_big_m": 50_000.0,  # MW – upper bound on electrolyser capacity
}
