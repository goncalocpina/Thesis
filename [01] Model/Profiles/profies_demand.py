'''Author: Gonçalo Costa Pina
Date_Created: 2026-02-21 (21st february 2026)
Date_Modified: 2026-02-21

----------------------------------------------

Plots demand profiles (Electricity, Heat, Gas) from CSV file
Data is NOT normalized - plotted as-is from the file
Note: The CSV file is expected to have a specific structure:  
    - time in the 3rd column and demand values in the 6th, 7th, and 8th columns for electricity, heat, and gas respectively; 
    - if the structure changes, the code will need to be updated accordingly.
'''

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np


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

base_dir = Path(__file__).resolve().parent

data_path = (
    base_dir.parent.parent
    / "[02] Data"
    / "Demand"
)

data_file = data_path / "Demand_Profiles.csv"


# =====================================================
# ==================== ELECTRICITY ====================
# =====================================================

df = pd.read_csv(data_file, sep=";", skiprows=0)
df_electricity = df[["Datetime (UTC)", "Rotterdam_electricity_load [MW]"]].copy()

df_electricity["electricity"] = pd.to_numeric(df_electricity["Rotterdam_electricity_load [MW]"], errors="coerce")
df_electricity = df_electricity.dropna(subset=["electricity"])

df_electricity["time"] = pd.to_datetime(df_electricity["Datetime (UTC)"], dayfirst=True)
df_electricity = df_electricity.sort_values("time")



# =====================================================
# ======================= HEAT ========================
# =====================================================

df = pd.read_csv(data_file, sep=";", skiprows=0)
df_heat = df[["Datetime (UTC)", "Rotterdam_total_heat_demand [MW]"]].copy()

df_heat["heat"] = pd.to_numeric(df_heat["Rotterdam_total_heat_demand [MW]"], errors="coerce")
df_heat = df_heat.dropna(subset=["heat"])

df_heat["time"] = pd.to_datetime(df_heat["Datetime (UTC)"], dayfirst=True)
df_heat = df_heat.sort_values("time")


# =====================================================
# ========================= H2 ========================
# =====================================================

df = pd.read_csv(data_file, sep=";", skiprows=0)
df_H2 = df[["Datetime (UTC)", "Rotterdam_total_gas_demand [MWh]"]].copy()

df_H2["h2"] = pd.to_numeric(df_H2["Rotterdam_total_gas_demand [MWh]"], errors="coerce")
df_H2 = df_H2.dropna(subset=["h2"])

df_H2["time"] = pd.to_datetime(df_H2["Datetime (UTC)"], dayfirst=True)
df_H2 = df_H2.sort_values("time")






# =====================================================
# ================= PLOTTING ==========================
# =====================================================


locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)


# =====================================================
# ================= Electricity =======================
# =====================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df_electricity["time"], df_electricity["electricity"])

#ax.set_ylim(0, 1)
ax.set_ylabel(r"Electricity Demand / [MWh]")
ax.set_xlabel(r"Date")

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Define Figures folder
figures_path = base_dir.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)  # create folder if it doesn't exist

# Define output file path
output_file = figures_path / "electricity_demand.pdf"

# Save figure
plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
print(f"Figure saved to: {output_file}")
plt.show()


# =====================================================
# ======================= Heat ========================
# =====================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df_heat["time"], df_heat["heat"])

#ax.set_ylim(0, 1)
ax.set_ylabel(r"Heat Demand / [MWh]")
ax.set_xlabel(r"Date")

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Define Figures folder
figures_path = base_dir.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)  # create folder if it doesn't exist

# Define output file path
output_file = figures_path / "heat_demand.pdf"

# Save figure
plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
print(f"Figure saved to: {output_file}")
plt.show()



# =====================================================
# ========================= H2 ========================
# =====================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df_H2["time"], df_H2["h2"])

#ax.set_ylim(0, 1)
ax.set_ylabel(r"H$_2$ Demand / [MWh]")
ax.set_xlabel(r"Date")

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Define Figures folder
figures_path = base_dir.parent.parent / "[03] Figures"
figures_path.mkdir(exist_ok=True)  # create folder if it doesn't exist

# Define output file path
output_file = figures_path / "H2_demand.pdf"

# Save figure
plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
print(f"Figure saved to: {output_file}")
plt.show()
