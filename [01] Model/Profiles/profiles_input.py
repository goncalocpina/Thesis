'''Author: Gonçalo Costa Pina
Date_Created: 2026-02-11 (11th february 2026)
Date_Modified: 2026-02-11

----------------------------------------------

Profile is normalized by the max value, not the sum of all values.
Profiles normalized by the sum are commented out in the end of this script.
Think if this should change

UPDATE: 
I think normalizing by max value is better bc in the model I will scale the profiles by 
capacity, and that  ultiplication will be limited to the capacity value. Or 1, 
depends on the scale.'''



from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import funcs as func


# =====================================================
# ========== TU Delft Thesis-Compatible Styling =======
# =====================================================

mpl.rcParams.update({

    # Do NOT require system LaTeX
    "text.usetex": False,
    "mathtext.fontset": "cm",

    # Serif font (LaTeX-like)
    "font.family": "serif",

    # Font sizes (match ~11pt thesis)
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # Axis style
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # Line width
    "lines.linewidth": 1.0,

    # Legend style
    "legend.frameon": False,

    # Export quality
    "figure.dpi": 300,
    "savefig.format": "pdf"
})


# =====================================================
# ==================== Paths ==========================
# =====================================================

DATA_DIR = Path(__file__).resolve().parent   # ← point to your directory


data_path = (DATA_DIR.parent.parent
    / "[02] Data"
    / "Supply")
print(f"Data path: {data_path}")

pv_file = data_path / "ninja_pv_51.9244_4.4778_corrected.csv"
wind_file = data_path / "ninja_wind_51.9244_4.4778_corrected.csv"


df_pv = pd.read_csv(pv_file, sep=",", skiprows=3)
df_pv = df_pv[["time", "electricity"]].copy()
df_pv["electricity"] = pd.to_numeric(df_pv["electricity"], errors="coerce")
df_pv = df_pv.dropna(subset=["electricity"])
df_pv["time"] = pd.to_datetime(df_pv["time"], format="%Y-%m-%d %H:%M")
df_pv = df_pv.sort_values("time")
norm_pv = df_pv["electricity"] 


df_wind = pd.read_csv(wind_file, sep=",", skiprows=3)
df_wind = df_wind[["time", "electricity"]].copy()
df_wind["electricity"] = pd.to_numeric(df_wind["electricity"], errors="coerce")
df_wind = df_wind.dropna(subset=["electricity"])
df_wind["time"] = pd.to_datetime(df_wind["time"], format="%Y-%m-%d %H:%M")
df_wind = df_wind.sort_values("time")
norm_wind= df_wind["electricity"]
print("PV max value:", norm_pv.max())
print("PV Capacity Factor: ", norm_pv.sum()/len(norm_pv))
print("PV Data Points: ", len(norm_pv))
print("Wind max value:", norm_wind.max())
print("Wind Capacity Factor: ", norm_wind.sum()/len(norm_wind))
print("Wind Number of values: ", len(norm_wind))
print("Wind Data Points: ", len(norm_wind))

# =====================================================
# ============ Common Date Formatting =================
# =====================================================

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)


# =====================================================
# ================= PV Plot ===========================
# =====================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df_pv["time"], norm_pv)

ax.set_ylim(0, 1)
ax.set_ylabel(r"Normalised PV production / [$-$]")
ax.set_xlabel(r"Date")

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Define Figures folder
figures_path = DATA_DIR.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)  # create folder if it doesn't exist

# Define output file path
pv_output_file = figures_path / "pv_supply_profile_normalized.pdf"

# Save figure
plt.tight_layout()
plt.savefig(pv_output_file, bbox_inches="tight")
print(f"Figure saved to: {pv_output_file}")
plt.show()


#Duration curve plot --------------------------------------------------
fig, axes = plt.subplots(figsize=(6, 4))

sorted_load = norm_pv.sort_values(ascending=False).values
axes.plot(sorted_load, color="steelblue", lw=1.5)
axes.set_xlabel("Hours per year (ranked)")
axes.set_ylabel(r"Normalised PV production / [$-$]")


# Define Figures folder and save
figures_path = DATA_DIR.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)

pv_duration_output_file = figures_path / "pv_duration_curve.pdf"

plt.tight_layout()
#plt.savefig(pv_duration_output_file, bbox_inches="tight")
#print(f"Figure saved to: {pv_duration_output_file}")
plt.show()

# =====================================================
# ================= Wind Plot =========================
# =====================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df_wind["time"], norm_wind)

ax.set_ylim(0, 1)
ax.set_ylabel(r"Normalised wind production / [$-$]")
ax.set_xlabel(r"Date")

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Define Figures folder
figures_path = DATA_DIR.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)  # create folder if it doesn't exist

# Define output file path
wind_output_file = figures_path / "wind_supply_profile_normalized.pdf"

# Save figure
plt.tight_layout()
plt.savefig(wind_output_file, bbox_inches="tight")
print(f"Figure saved to: {wind_output_file}")
plt.show()


#Duration curve plot --------------------------------------------------
fig, axes = plt.subplots(figsize=(6, 4))

sorted_load = norm_wind.sort_values(ascending=False).values
axes.plot(sorted_load, color="steelblue", lw=1.5)
axes.set_xlabel("Hours per year (ranked)")
axes.set_ylabel(r"Normalised Wind production / [$-$]")


# Define Figures folder and save
figures_path = DATA_DIR.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)

wind_duration_output_file = figures_path / "wind_duration_curve.pdf"

plt.tight_layout()
#plt.savefig(wind_duration_output_file, bbox_inches="tight")
#print(f"Figure saved to: {wind_duration_output_file}")
plt.show()