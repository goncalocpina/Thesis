import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Parameters
# =========================================================
Enom_batt = 100        # kWh nominal capacity
SOC_min = 0.2
SOC_max = 0.9
SoC_init = 0.5
SOH_init = 1.0
eta_batt = 0.85
dt_res = 1             # hours
Pnominv = 50           # kW inverter max
Life80_cal = 15         # calandric life indicator in years
Life80_cyc = 10000       # cycle life indicator in FEC
alpha_replace = 0.8    # minimum SOH

# =========================================================
# 2. Synthetic profiles
# =========================================================
timesteps = 48
hours = np.arange(timesteps)

# Day 1
P_PV_day1 = np.array([
    0,0,0,0,0,0,5,15,25,35,40,38,
    35,28,20,12,5,0,0,0,0,0,0,0])
P_Wind_day1 = np.array([
    12,12,16,15,16,18,20,22,18,17,15,14,
    12,13,15,18,20,22,21,19,16,14,13,12])

# Day 2
P_PV_day2 = np.array([
    0,0,0,0,0,0,3,10,20,28,32,30,
    28,22,15,10,4,0,0,0,0,0,0,0])
P_Wind_day2 = np.array([
    15,16,18,19,20,22,25,28,26,24,22,20,
    18,19,21,23,25,27,26,24,22,20,18,16])

P_PV = np.concatenate([P_PV_day1, P_PV_day2])
P_Wind = np.concatenate([P_Wind_day1, P_Wind_day2])
P_electrolyzer = np.full(timesteps, 25)

# =========================================================
# 3. Initialization
# =========================================================
E_useable = Enom_batt * (SOC_max - SOC_min)
SoC = np.zeros(timesteps)
E_batt = np.zeros(timesteps)
P_batt = np.zeros(timesteps)
SOH = np.zeros(timesteps)
aging_calendar = np.zeros(timesteps)
aging_cycle = np.zeros(timesteps)
aging_total = np.zeros(timesteps)

SoC[0] = SoC_init
E_batt[0] = SoC_init * E_useable
SOH[0] = SOH_init

# Counters for max 4-hour continuous charge/discharge
charge_hours = 0
discharge_hours = 0

# =========================================================
# 4. Simulation loop
# =========================================================
for t in range(1, timesteps):
    P_gen = P_PV[t] + P_Wind[t]
    P_load = P_electrolyzer[t]

    # Surplus or deficit
    P_surplus = max(P_gen - P_load, 0)
    P_deficit = max(P_load - P_gen, 0)

    # Maximum charge/discharge power based on SOC
    max_charge = min(P_surplus, (E_useable*SOC_max-E_batt[t-1])/dt_res, Pnominv)
    max_discharge = min(P_deficit, (E_batt[t-1]-E_useable*SOC_min)/dt_res, Pnominv)

    # Apply 4-hour continuous limit
    if charge_hours >= 4:
        max_charge = 0
    if discharge_hours >= 4:
        max_discharge = 0

    # Decide battery action
    if max_charge > 0:
        P_batt[t] = max_charge
        charge_hours += 1
        discharge_hours = 0
    elif max_discharge > 0:
        P_batt[t] = -max_discharge
        discharge_hours += 1
        charge_hours = 0
    else:
        P_batt[t] = 0
        charge_hours = 0
        discharge_hours = 0

    # Energy update
    if P_batt[t] > 0:  # charging
        delta_E = P_batt[t]/eta_batt*dt_res
    else:  # discharging
        delta_E = P_batt[t]*eta_batt*dt_res

    E_batt[t] = np.clip(E_batt[t-1] + delta_E, 0, E_useable)
    SoC[t] = E_batt[t]/E_useable

    # ===================
    # Battery ageing
    # ===================
    # Calendar ageing
    aging_calendar[t] = aging_calendar[t-1] + dt_res / (Life80_cal * 8760)
    # Cycle ageing
    if E_batt[t-1] > 0:
        aging_cycle[t] = aging_cycle[t-1] + abs(P_batt[t]) * dt_res / E_useable / Life80_cyc
    else:
        aging_cycle[t] = aging_cycle[t-1]


    # Total ageing
    aging_total[t] = aging_calendar[t] + aging_cycle[t]
    # SOH update
    SOH[t] = max(alpha_replace, SOH[t-1]-0.2*(aging_total[t]-aging_total[t-1]))
    
    # ================================================
    # Battery replacement estimation
    # ================================================

    def estimate_battery_replacement(SOH_array, dt_res=1):
        replacement_index = np.where(SOH_array <= 0.8)[0]

        if len(replacement_index) > 0:
            hours_to_replace = replacement_index[0] * dt_res
            return hours_to_replace, replacement_index[0]
        
        else:
        # Extrapolate if SOH doesn't reach 0.8 in this simulation
            total_hours = len(SOH_array) * dt_res
            delta_SOH = SOH_array[0] - SOH_array[-1]
            avg_decay_per_hour = delta_SOH / total_hours

        if avg_decay_per_hour > 0:
            hours_to_replace = (SOH_array[-1] - 0.8) / avg_decay_per_hour + total_hours
            return hours_to_replace, None
        else:
            return None, None

hours_to_replace, idx_replace = estimate_battery_replacement(SOH, dt_res)

print(f"Estimated battery replacement after {hours_to_replace:.1f} hours (~{hours_to_replace/8760:.1f} years)")


# =========================================================
# 5. Plot results
# =========================================================
plt.figure(figsize=(12,12))

plt.subplot(5,1,1)
plt.plot(P_PV, label="PV Power")
plt.plot(P_Wind, label="Wind Power")
plt.plot(P_electrolyzer, label="Electrolyzer Load")
plt.ylabel("kW"); plt.legend(); plt.grid(True)

plt.subplot(5,1,2)
plt.plot(P_batt, label="Battery Power", color='purple')
plt.ylabel("kW"); plt.legend(); plt.grid(True)

plt.subplot(5,1,3)
plt.plot(SoC, label="SoC")
plt.ylabel("SOC (pu)"); plt.legend(); plt.grid(True)

plt.subplot(5,1,4)
plt.plot(E_batt, label="Battery Energy (kWh)", color='orange')
plt.ylabel("Energy (kWh)"); plt.legend(); plt.grid(True)

plt.subplot(5,1,5)
plt.plot(SOH, label="State of Health (SOH)", color='green')
plt.ylim(0.9998, 1.00)  # zoom op de kleine verandering
plt.ylabel("SOH (pu)"); plt.xlabel("Time (h)"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()