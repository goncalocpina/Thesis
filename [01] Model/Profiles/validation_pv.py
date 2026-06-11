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

pv_file = data_path / "ninja_pv_51.9244_4.4778_corrected.csv"


# =====================================================
# ================= Load Data ==========================
# =====================================================

df_pv = pd.read_csv(pv_file, sep=",", skiprows=3)

df_pv = df_pv[["time", "electricity", "irradiance_direct", "irradiance_diffuse"]].copy()

for col in ["electricity", "irradiance_direct", "irradiance_diffuse"]:
    df_pv[col] = pd.to_numeric(df_pv[col], errors="coerce")

df_pv = df_pv.dropna(subset=["electricity", "irradiance_direct", "irradiance_diffuse"])

df_pv["time"] = pd.to_datetime(df_pv["time"], format="%Y-%m-%d %H:%M")
df_pv = df_pv.sort_values("time").set_index("time")


# =====================================================
# ================= PV Model ======================
# =====================================================

P_max = 1000            # Wp per panel
eta_module = 0.255

# Define number of panels (you MUST set this)
n_panels = 1   # <-- CHANGE THIS to your real value

# Panel area (m²)
panel_area = P_max / (eta_module * 1000)  # A = P / (eta * 1000 W/m²)

# POA irradiance (W/m²)
df_pv["G_poa"] = df_pv["irradiance_direct"] + df_pv["irradiance_diffuse"]

# Physical PV per panel (kW)
df_pv["power_calc"] = df_pv["G_poa"]  *0.9 # W → kW

# RN per panel (assuming electricity is total system kW)
df_pv["power_given"] = df_pv["electricity"] 


# =====================================================
# ============ Select One Day ==========================
# =====================================================

chosen_date = "2019-06-01" # try "2023-07-01", "2023-01-01" and "2023-11-06"
start = pd.to_datetime(chosen_date)
end = start + pd.Timedelta(days=1)

# Select full day INCLUDING next midnight
df_pv_day = df_pv.loc[start:end].copy()

print("\nSelected day data:")
print(df_pv_day)
print("Max calculated power:", df_pv_day["power_calc"].max())
print("Max given power:", df_pv_day["power_given"].max())


# =====================================================
# ============ Common Date Formatting =================
# =====================================================

locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
formatter = mdates.DateFormatter("%H:%M")


# =====================================================
# ================= Plot ==============================
# =====================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df_pv_day.index,
        df_pv_day["power_given"],
        label="Given PV power")

ax.plot(df_pv_day.index,
        df_pv_day["power_calc"],
        linestyle="--",
        label="Calculated power")

ax.set_ylabel(r"PV power [-]")
# Convert to datetime object
dt = datetime.strptime(chosen_date, "%Y-%m-%d")
# Format as "Month day" (e.g., "November 6")
formatted_date = dt.strftime("%B %#d") 
ax.set_xlabel(f"Time of day ({formatted_date})")
#ax.set_xlabel(f"Time of day ({chosen_date})")

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
#ax.set_xlim(df_pv_day.index.min(), df_pv_day.index.max())

ax.legend(loc="best")   # <-- add this line

'''# Convert to datetime object
dt = datetime.strptime(chosen_date, "%Y-%m-%d")
# Format as "Month day" (e.g., "November 6")
formatted_date = dt.strftime("%B %#d") 
ax.set_xlabel(f"Time of day ({formatted_date})")
'''

# =====================================================
# ================= Save Figure =======================
# =====================================================

figures_path = DATA_DIR.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)

output_file = figures_path / "pv_power_validation.pdf"

plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
print(f"Figure saved to: {output_file}")
plt.show()