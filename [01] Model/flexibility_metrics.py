# flexibility_metrics.py
"""
Flexibility quantification for the sector-coupled electricity–heat–H₂ model.
All functions receive pandas Series with a DatetimeIndex at hourly resolution.

Flexibility categories:
  1. Storage-based     : battery, salt cavern, thermal storage
  2. Sector-coupling   : electrolyser, heat pump, CHP
  3. Demand-side       : curtailment (proxy for flexibility failure)
  4. Ramping           : residual load ramps on the electricity bus
  5. System-level      : self-sufficiency, sector coupling ratios
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. STORAGE-BASED FLEXIBILITY
# =============================================================================

def storage_utilisation_rate(soc: pd.Series, capacity: float) -> float:
    """
    Average fraction of storage capacity that is occupied.
    Analogous to a capacity factor for energy storage.
    
    Formula: mean(SoC[t]) / capacity
    Range  : 0–1  (1 = always full, 0 = always empty)

    Question: What is the optinal utilisation rate for flexibility? Somewhere in the range 0.3 - 0.7?
    """
    if capacity <= 0:
        return 0.0
    return float(soc.mean() / capacity)


def throughput_ratio(charge: pd.Series, capacity: float) -> float:
    """
    Annual energy cycled through storage relative to its capacity.
    Indicates how many times the storage is fully cycled per year.
    
    Formula: sum(charge[t]) / capacity
    Unit   : cycles/year
    """
    if capacity <= 0:
        return 0.0
    return float(charge.sum() / capacity)


def load_shift_index(charge: pd.Series, capacity: float) -> float: #FIXME: This one is the same as above?
    """
    I_shift: thermal load shifted relative to storage size.
    Directly from the KTH district heating paper (Calvo García et al., 2025).
    Higher values = storage is more intensively used relative to its size.
    
    Formula: sum(|charge[t]|) / capacity
    """
    if capacity <= 0:
        return 0.0
    return float(charge.abs().sum() / capacity)


def load_shift_index(
    discharge: pd.Series,
    soc: pd.Series,
    capacity: float,
    power_rating: float,
    min_soc_fraction: float = 0.0,
    dt: float = 1.0,
) -> float:
    """
    Discharge Efficiency Index: ratio of actual energy discharged to the
    maximum energy that *could* have been discharged during the same hours.

    For each timestep t where discharge > 0, the maximum dischargeable energy is:

        max_discharge[t] = min(
            power_rating * dt,                          # power constraint
            (soc[t] - min_soc_abs) * outflow_factor     # energy constraint
        )

    where min_soc_abs = min_soc_fraction * capacity.

    The index is:
        I = sum(discharge[t]) / sum(max_discharge[t])    ∈ [0, 1]

    A value of 1.0 means: whenever the storage discharged, it did so at the
    maximum rate physically possible. A low value means it discharged
    conservatively relative to what was available.

    Parameters
    ----------
    discharge         : hourly discharge flow [MWh/h]
    soc               : hourly state of charge [MWh], aligned with discharge
    capacity          : storage energy capacity [MWh]
    power_rating      : maximum discharge power [MW]  (= MW, since dt=1h → MWh/h)
    min_soc_fraction  : minimum SoC as fraction of capacity (e.g. 0.2 for 20%)
    dt                : timestep duration [h], default 1
    """
    if capacity <= 0 or power_rating <= 0:
        return 0.0

    min_soc_abs = min_soc_fraction * capacity

    # Only consider timesteps where actual discharge occurred
    active = discharge > 0

    if not active.any():
        return 0.0

    # Available energy above minimum SoC at each active timestep
    # Note: soc[t] is the SoC at the *start* of interval t (oemof convention)
    available_energy = (soc[active] - min_soc_abs).clip(lower=0.0)

    # Maximum discharge is the tighter of the two constraints
    max_discharge = np.minimum(
        power_rating * dt,       # power rating constraint
        available_energy,        # SoC constraint
    )

    total_max = max_discharge.sum()
    if total_max <= 0:
        return 0.0

    return float(discharge[active].sum() / total_max)


def storage_flexibility_band(soc: pd.Series, capacity: float) -> dict:
    """
    At each hour, computes:
      - upward flexibility   = remaining headroom (can absorb more)
      - downward flexibility = current SoC (can release this much)
    Returns annual averages of both.
    
    Unit: MWh (average available flexibility at any given hour)
    """
    if capacity <= 0:
        return {"upward_mwh": 0.0, "downward_mwh": 0.0}
    upward   = capacity - soc          # space to charge
    downward = soc                     # energy available to discharge
    return {
        "upward_mwh":   float(upward.mean()),
        "downward_mwh": float(downward.mean()),
    }


def soc_seasonal_profile(soc: pd.Series) -> pd.DataFrame:
    """
    Returns monthly average SoC, useful for identifying seasonal patterns
    (e.g., salt cavern charging in summer, discharging in winter).
    """
    return soc.resample("ME").mean().rename("mean_soc")


# =============================================================================
# 2. SECTOR-COUPLING FLEXIBILITY
# =============================================================================

def sector_coupling_ratio(
    exchange_mwh: float,
    total_supply_demand_mwh: float,
    efficiency: float = 1.0,
    with_losses: bool = False,
) -> float:
    """
    Energy exchange ratio between sectors, from PATHFNDR (ETH Zurich, 2023).
    
    Without losses: exchange / total_supply_demand
    With losses   : (exchange × efficiency) / total_supply_demand
    
    Returns a dimensionless ratio in [0, 1].
    """
    if total_supply_demand_mwh <= 0:
        return 0.0
    if with_losses:
        return float((exchange_mwh * efficiency) / total_supply_demand_mwh)
    return float(exchange_mwh / total_supply_demand_mwh)


def electrolyser_flexibility_metrics(
    el_input: pd.Series,
    el_output_h2: pd.Series,
    el_output_heat: pd.Series,
    capacity: float,
    total_el_mwh: float,
    total_h2_mwh: float,
    efficiency_h2: float,
    efficiency_heat: float,
) -> dict:
    """
    Computes all flexibility metrics for the electrolyser:
    - Capacity factor (how often it runs vs installed capacity)
    - Load factor std (variability of operation — high std = more flexible dispatch)
    - Sector coupling ratio (electricity → H₂)
    - Sector coupling ratio with losses
    - Power-to-Heat fraction (share of output that is heat)
    - Ramp rates (MW/h, up and down)
    """
    if capacity <= 0:
        return {}

    cf = float(el_input.mean() / capacity)

    # Normalised load = fraction of capacity used each hour
    load_normalised = el_input / capacity
    load_std        = float(load_normalised.std())  #Calculates std deviation, high = more variable dispatch

    # Ramp rates (hour-to-hour change in input power)
    ramps       = el_input.diff().dropna()
    ramp_up     = float(ramps[ramps > 0].mean())   # MW/h upward
    ramp_down   = float(ramps[ramps < 0].abs().mean())  # MW/h downward

    sc_ratio          = sector_coupling_ratio(el_input.sum(), total_el_mwh)
    sc_ratio_with_loss = sector_coupling_ratio(el_input.sum(), total_el_mwh,
                                               efficiency=efficiency_h2, with_losses=True)

    # Hours at minimum load vs zero (start-stop behaviour)
    hours_off         = int((el_input == 0).sum())
    hours_at_min      = int((el_input > 0).sum())

    return {
        "capacity_factor":        cf,
        "load_variability_std":   load_std,
        "ramp_up_avg_mw_per_h":   ramp_up,
        "ramp_down_avg_mw_per_h": ramp_down,
        "sc_ratio_el_to_h2":      sc_ratio,
        "sc_ratio_with_losses":   sc_ratio_with_loss,
        "ptx_heat_fraction":      float(el_output_heat.sum() / (el_output_h2.sum() + el_output_heat.sum())) if (el_output_h2.sum() + el_output_heat.sum()) > 0 else 0.0, ##FIXME: wrong formula
        "hours_offline":          hours_off,
        "hours_operating":        hours_at_min,
    }


def chp_flexibility_metrics(
    h2_input: pd.Series,
    el_output: pd.Series,
    heat_output: pd.Series,
    capacity: float,
    total_h2_mwh: float,
) -> dict:
    """
    CHP (fuel cell) flexibility metrics:
    - Sector coupling ratio (H₂ → electricity + heat)
    - Power-to-heat ratio variability
    - Ramp rates
    """
    if capacity <= 0:
        return {}

    cf          = float(h2_input.mean() / capacity)
    ramps       = h2_input.diff().dropna()
    ramp_up     = float(ramps[ramps > 0].mean())
    ramp_down   = float(ramps[ramps < 0].abs().mean())

    total_output = el_output.sum() + heat_output.sum()
    el_frac      = float(el_output.sum() / total_output) if total_output > 0 else 0.0
    heat_frac    = float(heat_output.sum() / total_output) if total_output > 0 else 0.0

    return {
        "capacity_factor":        cf,
        "ramp_up_avg_mw_per_h":   ramp_up,
        "ramp_down_avg_mw_per_h": ramp_down,
        "sc_ratio_h2_consumed":   sector_coupling_ratio(h2_input.sum(), total_h2_mwh), #FIXME: wrong formula
        "output_el_fraction":     el_frac,
        "output_heat_fraction":   heat_frac,
    }


# =============================================================================
# 3. DEMAND-SIDE FLEXIBILITY (curtailment as a proxy for flexibility failure)
# =============================================================================

def curtailment_metrics(
    curtailment: pd.Series,
    total_generation: float,
    label: str = "electricity",
) -> dict:
    """
    Curtailment = flexibility failure: generation could not be absorbed.
    
    - curtailment_mwh       : total energy wasted
    - curtailment_fraction  : fraction of total generation curtailed
    - curtailment_hours     : number of hours with any curtailment
    - peak_curtailment_mw   : worst single-hour curtailment
    """
    total_curt = float(curtailment.sum())
    return {
        f"{label}_curtailment_mwh":      total_curt,
        f"{label}_curtailment_fraction": total_curt / total_generation if total_generation > 0 else 0.0,
        f"{label}_curtailment_hours":    int((curtailment > 0).sum()),
        f"{label}_peak_curtailment_mw":  float(curtailment.max()),
    }


# =============================================================================
# 4. RAMPING FLEXIBILITY (electricity bus residual load)
# =============================================================================

def residual_load(
    consumption: pd.Series,
    generation: pd.Series,
) -> pd.Series:
    """
    Residual load = demand after subtracting variable renewables.
    Positive = must be covered by dispatchable assets or storage.
    Negative = surplus renewable generation (over-generation situation).
    """
    return consumption - generation


def ramping_metrics(res_load: pd.Series) -> dict:
    """
    Computes ramp statistics on the residual load timeseries.
    Based on ENTSO-E (2021) and PATHFNDR methodology.
    
    - ramp_1h  : max 1-hour ramp (MW)
    - ramp_3h  : max 3-hour ramp (MW)
    - ramp_8h  : max 8-hour ramp (MW)
    - ramp_std : standard deviation of 1-hour ramps (flexibility pressure)
    - over_gen_hours : hours of negative residual load (surplus)
    - over_gen_ratio : peak / minimum residual load
    """
    ramp_1h = res_load.diff(1).abs()
    ramp_3h = res_load.diff(3).abs()
    ramp_8h = res_load.diff(8).abs()

    over_gen_hours = int((res_load < 0).sum())
    rl_range       = float(res_load.max() - res_load.min())

    return {
        "ramp_1h_max_mw":     float(ramp_1h.max()),
        "ramp_1h_p95_mw":     float(ramp_1h.quantile(0.95)),
        "ramp_3h_max_mw":     float(ramp_3h.max()),
        "ramp_8h_max_mw":     float(ramp_8h.max()),
        "ramp_std_mw":        float(ramp_1h.std()),
        "over_gen_hours":     over_gen_hours,
        "over_gen_fraction":  over_gen_hours / len(res_load),
        "residual_load_range_mw": rl_range,
    }


# =============================================================================
# 5. SYSTEM-LEVEL FLEXIBILITY
# =============================================================================

def self_sufficiency_ratio(
    demand: pd.Series,
    local_generation: pd.Series,
) -> float:
    """
    SSR = fraction of demand met by local generation (without imports).
    Based on Pastore (2026), Sustainability 18, 437.

    Self-Sufficiency Ratio (SSR): What fraction of demand is met directly by local generation. 
    SSR = 0.4 means 40% of electricity demand is met instantaneously by renewables without storage. 
    The remaining 60% requires either stored energy or sector-coupled reconversion (CHP).
    
    Formula: sum(min(gen[t], demand[t])) / sum(demand[t])
    """
    self_consumed = np.minimum(local_generation.values, demand.values)
    return float(self_consumed.sum() / demand.sum()) if demand.sum() > 0 else 0.0


def self_consumption_ratio(
    demand: pd.Series,
    local_generation: pd.Series,
) -> float:
    """
    SCR = fraction of local generation that is consumed locally.

    Self-Consumption Ratio (SCR): What fraction of generation is consumed directly on site. 
    SCR = 0.6 means 60% of renewable generation goes directly to meeting demand. 
    The remaining 40% either goes into storage, the electrolyser, the heat pump, or is curtailed.
    
    Formula: sum(min(gen[t], demand[t])) / sum(gen[t])
    """
    self_consumed = np.minimum(local_generation.values, demand.values)
    return float(self_consumed.sum() / local_generation.sum()) if local_generation.sum() > 0 else 0.0


def flexibility_summary_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Converts the nested metrics dictionary into a flat DataFrame
    suitable for display, CSV export, or radar chart input.
    """
    rows = []
    for category, metrics in metrics_dict.items():
        for metric, value in metrics.items():
            rows.append({"category": category, "metric": metric, "value": value})
    return pd.DataFrame(rows)