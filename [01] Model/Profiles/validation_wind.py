'''Author: Gonçalo Costa Pina
Date_Created: 2026-03-17
Date_Modified: 2026-03-17
'''

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np
from datetime import datetime

# =====================================================
# ========== TU Delft Thesis-Compatible Styling =======
# =====================================================

mpl.rcParams.update({

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
    "savefig.format": "pdf"
})


# =====================================================
# ==================== Paths ==========================
# =====================================================

DATA_DIR = Path(__file__).resolve().parent

data_path = (DATA_DIR.parent.parent
    / "[02] Data"
    / "Supply")
print(f"Data path: {data_path}")

wind_file = data_path / "ninja_wind_51.9244_4.4778_corrected.csv"


# =====================================================
# ================= Load Data ==========================
# =====================================================

df_wind = pd.read_csv(wind_file, sep=",", skiprows=3)

df_wind = df_wind[["time", "electricity", "wind_speed"]].copy()

df_wind["electricity"] = pd.to_numeric(df_wind["electricity"], errors="coerce")
df_wind["wind_speed"] = pd.to_numeric(df_wind["wind_speed"], errors="coerce")

df_wind = df_wind.dropna(subset=["electricity", "wind_speed"])

df_wind["time"] = pd.to_datetime(df_wind["time"], format="%Y-%m-%d %H:%M")
df_wind = df_wind.sort_values("time").set_index("time")


# =====================================================
# ================= Turbine Model ======================
# =====================================================
#Source:
# https://en.wind-turbine-models.com/turbines/16-vestas-v90
eta = 0.5926
rho = 1.225
rotor_r = 45
A = np.pi * rotor_r**2

v_cut_in = 3
v_rated = 12
v_cut_out = 25
P_rated = 2000  # kW


def calc_power(v):
    if v < v_cut_in or v >= v_cut_out:
        return 0
    elif v < v_rated:
        return min(eta * 0.5 * rho * A * v**3 / 1000/ P_rated, 1)
    else:
        return 1


# =====================================================
# ============ Compute Power ===========================
# =====================================================

df_wind["power_calc"] = df_wind["wind_speed"].apply(calc_power)


# =====================================================
# ============ Select One Day ==========================
# =====================================================

chosen_date = "2019-06-01" # try "2023-07-01", "2023-01-01" and "2023-11-06"
start = pd.to_datetime(chosen_date)
end = start + pd.Timedelta(days=1)
print("Max calculated power:", df_wind["power_calc"].max())
print("Max given power:", df_wind["electricity"].max())
# Select full day INCLUDING next midnight
df_wind_day = df_wind.loc[start:end].copy()

print("\nSelected day data:")
print(df_wind_day)


# =====================================================
# ============ Common Date Formatting =================
# =====================================================

locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
formatter = mdates.DateFormatter("%H:%M")


# =====================================================
# ================= Plot ==============================
# =====================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df_wind_day.index,
        df_wind_day["electricity"],
        label="Given wind power")

ax.plot(df_wind_day.index,
        df_wind_day["power_calc"],
        linestyle="--",
        label="Calculated power")

ax.set_ylabel(r"Wind power [-]")

#ax.set_xlabel(f"Time of day ({chosen_date})")
# Convert to datetime object
dt = datetime.strptime(chosen_date, "%Y-%m-%d")
# Format as "Month day" (e.g., "November 6")
formatted_date = dt.strftime("%B %#d") 
ax.set_xlabel(f"Time of day ({formatted_date})")
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)


ax.legend(loc="upper center")   # <-- add this line


# =====================================================
# ================= Save Figure =======================
# =====================================================

figures_path = DATA_DIR.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)

output_file = figures_path / "wind_power_validation.pdf"

plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
print(f"Figure saved to: {output_file}")
plt.show()