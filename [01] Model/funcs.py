'''Author: Gonçalo Costa Pina
Date_Created: 2026-03-26 (26th March 2026)
Date_Modified: 2026-03-26

----------------------------------------------

Defines functions used across the project
'''
import matplotlib as mpl
import numpy as np
import pandas as pd
from Parameters import parameters as par
from scipy.optimize import brentq
import matplotlib.ticker as mticker
import numpy_financial as npf
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from oemof.tools import economics
import pyomo.environ as po


#Techno-economic parameters   
wacc = par.WACC                    # 7%
lifetime = par.LIFETIME            # years
net_profit_margin = par.MARGIN     # 8.1% 


#──────────────────────────────────────────────────────────────────────────────
# SAVE FIGURE FUNCTION
#──────────────────────────────────────────────────────────────────────────────

def savefigure(fig, filename, ext="pdf"):
    """
    Save a matplotlib figure to the [03] Figures folder.

    Parameters:
        fig      : matplotlib figure object
        filename : name of the file (without extension)
        ext      : file extension (default: pdf)
    """
    base_dir = Path(__file__).resolve().parent          # .../PROJECT_MAIN/[01] Model
    figures_path = base_dir.parent / "[03] Figures"     # .../PROJECT_MAIN/[03] Figures
    figures_path.mkdir(exist_ok=True)

    output_file = figures_path / f"{filename}.{ext}"

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")

    print(f"Figure saved to: {output_file}")


#──────────────────────────────────────────────────────────────────────────────
# SYSTEM VISUALIZATION HELPER FUNCTION
#──────────────────────────────────────────────────────────────────────────────

def push_apart(pos, radius, iterations=300, strength=0.5):
    pos = {n: np.array(p, dtype=float) for n, p in pos.items()}
    nodes = list(pos.keys())
    for _ in range(iterations):
        moved = False
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                delta = pos[u] - pos[v]
                dist  = np.linalg.norm(delta)
                min_dist = 2 * radius
                if dist < min_dist and dist > 1e-6:
                    push  = (delta / dist) * (min_dist - dist) * strength
                    pos[u] += push
                    pos[v] -= push
                    moved = True
        if not moved:
            break
    return pos


def to_pyvis_coords(pos, scale=600):
    """Normalise a networkx pos dict to pyvis pixel coordinates."""
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    result = {}
    for node, (x, y) in pos.items():
        px =  (x - x_min) / (x_max - x_min) * scale - scale / 2
        py = -(y - y_min) / (y_max - y_min) * scale + scale / 2  # flip y
        result[node] = (px, py)
    return result


def classify_node(label):
    label = str(label).lower()
    if label in ('electricity', 'hydrogen', 'heat'):
        return 'bus'
    if 'curtail' in label:
        return 'curtailment'
    if 'demand' in label:
        return 'demand'
    if label in ('battery', 'saltcavern', 'thermal_storage'):
        return 'storage'
    if label in ('pv', 'wind'):
        return 'supply'
    return 'ptx'  # electrolyser, heatpump, chp


def build_sankey_flows(flow_specs):
    """flow_specs: list of (source_idx, target_idx, value_series_or_scalar)"""
    src, tgt, val, lbl = [], [], [], []
    for s, t, v_raw, label in flow_specs:
        v = float(v_raw.sum()) if hasattr(v_raw, "sum") else float(v_raw)
        if v > 0.1:          # suppress near-zero flows
            src.append(s); tgt.append(t)
            val.append(v / 1e3)   # MWh → GWh
            lbl.append(label)
    return src, tgt, val, lbl


#──────────────────────────────────────────────────────────────────────────────
# COMBINING TIME SERIES + DURATION CURVE PLOT FUNCTION
#──────────────────────────────────────────────────────────────────────────────

_abc_labels = ["(a)", "(b)", "(c)"]   # private to this module, no need to pass in

def plot_twinxy_row(ax_left, flow, capacity, cf, label, color, is_storage, row_index, n_rows):
    abc = _abc_labels[row_index]

    n_hours     = len(flow)
    x_norm      = np.linspace(0, 1, n_hours)
    sorted_vals = flow.sort_values(ascending=False).values

    ax_dur = ax_left.twiny()
    ax_dur.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # ── Top x-axis: only show on first row ───────────────────────────────────
    if row_index == 0:
        ax_dur.tick_params(axis="x", labeltop=True)
    else:
        ax_dur.tick_params(axis="x", labeltop=False, top=False)

    if is_storage:
        ax_left.fill_between(flow.index, flow.values, 0,
                             color=color, alpha=0.35, zorder=1, label="SOC")
    else:
        ax_left.plot(flow.index, flow.values,
                     color=color, lw=0.8, zorder=1, label="Load")

    if capacity > 0:
        ax_left.axhline(capacity, color="black", ls="--", lw=1,
                        label="Installed capacity", zorder=2)

    ax_left.set_ylabel(label)
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # ── Bottom x-axis: only show on last row ──────────────────────────────────
    if row_index == n_rows - 1:
        ax_left.tick_params(axis="x", labelbottom=True)
        ax_left.set_xlabel("Month")
    else:
        ax_left.tick_params(axis="x", labelbottom=False, bottom=False)

    ax_dur.plot(x_norm, sorted_vals,
                color="black", lw=1, zorder=4, label="Duration curve")

    dc_label = f"Average SOC: {cf:.2f}" if is_storage else f"CF: {cf:.2f}"
    if capacity > 0:
        ax_dur.axhline(cf * capacity, color="black", ls=":", lw=1,
                       zorder=4, label=dc_label)

    ax_left.text(-0.1, 1, abc,
                 transform=ax_left.transAxes,
                 fontsize=18, fontweight="bold", va="top", ha="center", zorder=10)

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_dur.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   borderaxespad=0, frameon=True, fontsize=20)


def plot_combined_twinxy(data_list, is_storage, fname):
    n = len(data_list)

    with mpl.rc_context({

    # Do NOT require system LaTeX
    "text.usetex": False,
    "mathtext.fontset": "cm",

    # Serif font (LaTeX-like)
    "font.family": "serif",

    # Font sizes (match ~11pt thesis)
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,

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
    }):
        fig, axes = plt.subplots(n, 1, figsize=(15.5, 4.5 * n), sharex=False)
        if n == 1:
            axes = [axes]

        for row, (label, flow, capacity, cf, color) in enumerate(data_list):
            plot_twinxy_row(axes[row], flow, capacity, cf,
                            label, color, is_storage, row, n)  # pass n here
        
        plt.tight_layout(rect=[0, 0, 0.92, 1])
        savefigure(fig, fname)
        plt.show()


#──────────────────────────────────────────────────────────────────────────────
# LEVELISED COSTS FIGURE FUNCTION
#──────────────────────────────────────────────────────────────────────────────

def plot_lc_and_demand(lc_ts, demand_ts, annual_lc,
                       lc_label, demand_label, lc_color, demand_color,
                       fig_name):
    """Top panel: instantaneous LC + annual average line.
       Bottom panel: corresponding demand timeseries. Shared x-axis."""

    with mpl.rc_context({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.format": "pdf",
    }):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.subplots_adjust(hspace=0.08)

        axes[0].plot(lc_ts.index, lc_ts.values,
                     color=lc_color, linewidth=0.5,
                     label=f"Instantaneous {lc_label}")
        axes[0].axhline(annual_lc, color="black", linewidth=1.2, linestyle="--",
                        label=f"Annual avg = {annual_lc:.1f} €/MWh")
        axes[0].set_ylabel(f"{lc_label}")
        axes[0].legend(loc="best", fontsize=15)

        axes[1].plot(demand_ts.index, demand_ts.values,
                     color=demand_color, alpha=0.6, label=demand_label)
        axes[1].set_ylabel(f"{demand_label} / [MWh/h]")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="best", fontsize=15)

        axes[1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        plt.tight_layout()
        savefigure(fig, fig_name)
        plt.show()

#──────────────────────────────────────────────────────────────────────────────
# CREATE AND SAVE LOGBOOK
#──────────────────────────────────────────────────────────────────────────────
    
def save_logbook_csv(logbook: dict, data_dir, scenario_name: str):
    """
    Append one row of scalar results to a growing CSV logbook.
    File: [data_dir]/Logbooks/Logbook_{SCENARIO_NAME}.csv
    One row per simulation run, one column per variable.
    """
    import csv
    from datetime import datetime
    from pathlib import Path

    # Ensure the Logbooks folder exists
    logbooks_path = Path(data_dir) / "Logbooks"
    logbooks_path.mkdir(exist_ok=True)

    # CSV path with scenario name
    safe_scenario_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in scenario_name)
    csv_path = logbooks_path / f"Logbook_{safe_scenario_name}.csv"

    # Flatten the logbook and add timestamp
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update(_flatten_dict(logbook))

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logbook row appended to: {csv_path}")


def _flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten nested dict for CSV export."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


#──────────────────────────────────────────────────────────────────────────────
# OPERATIONAL METRICS FUNCTIONS 
#──────────────────────────────────────────────────────────────────────────────

def capacity_factor(series_power, nominal_capacity, timestep_hours):
    if nominal_capacity == 0:
        return 0
    total_energy = series_power.sum() * timestep_hours      # kWh
    max_energy = nominal_capacity * len(series_power) * timestep_hours  # kWh
    return total_energy / max_energy

def average_power(series_power):
    return series_power.mean()


#──────────────────────────────────────────────────────────────────────────────
# TOTAL COSTS BREAKDOWN HELPER FUNCTION 
#──────────────────────────────────────────────────────────────────────────────

def sorted_pie(labels_vals):
    pairs = [(l, v) for l, v in labels_vals if v > 0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return zip(*pairs) if pairs else ([], [])


def get_color(COMP_ALIASES, COMP_COLORS, comp):
    base = COMP_ALIASES.get(comp, comp)
    return COMP_COLORS.get(base, "grey")

#──────────────────────────────────────────────────────────────────────────────
# REPLACEMENT COSTS HELPER FUNCTIONS 
#──────────────────────────────────────────────────────────────────────────────

# ── Helper: split annualised cost into "first unit" and "replacements" ────────
def split_cost(capex, size, n, wacc, u):
    """
    Returns (cost_first, cost_replacements) in €/yr.

    - cost_first        : annuity(capex, n, wacc, u) * (u/n) * size
                          → the fraction of the annualised cost corresponding
                            to the first unit (years 0 to u)
    - cost_replacements : annuity(capex, n, wacc, u) * ((n-u)/n) * size
                          → the remaining fraction, corresponding to all
                            replacement purchases (years u to n)

    For permanent parts (no replacement), pass u=n so cost_replacements=0.
    """
    a     = economics.annuity(capex=capex, n=n, wacc=wacc, u=u)
    first = a * (u / n) * size
    repl  = a * ((n - u) / n) * size
    return first, repl


def cost_permanent(capex, size, n, wacc):
    """
    Returns the annualised cost of a truly permanent part (no replacement).
    Entirely attributed to the bottom bar.
    """
    return economics.annuity(capex=capex, n=n, wacc=wacc) * size


#──────────────────────────────────────────────────────────────────────────────
# BAR PLOTS HELPER FUNCTIONS 
#──────────────────────────────────────────────────────────────────────────────


def safeval(val, default=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)

# Helper: annotate bar top and (optionally) inner segment
def annotate_top(ax, x, total, unit=""):
    ax.text(x, total + ax.get_ylim()[1] * 0.01,
            f"{total:.1f}{unit}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

def annotate_inner(ax, x, y_centre, text):
    """Only annotate if segment is tall enough to fit text."""
    if y_centre > ax.get_ylim()[1] * 0.03:
        ax.text(x, y_centre, text,
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")
        



#──────────────────────────────────────────────────────────────────────────────
# RADAR PLOTS HELPER FUNCTIONS 
#──────────────────────────────────────────────────────────────────────────────

def draw_radar(ax, data: dict, colors: list, abc: str = ""):
    categories = list(list(data.values())[0].keys())
    N          = len(categories)
    angles     = [n / float(N) * 2 * np.pi for n in range(N)]
    angles    += angles[:1]

    for (tech, vals), color in zip(data.items(), colors):
        values  = [vals[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, lw=1.8, color=color, label=tech)
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])

    # ── Force category labels outside the chart ───────────────────────────────
    ax.set_xticklabels([])   # clear default labels
    for angle, label in zip(angles[:-1], categories):
        ha = "center"
        if angle < np.pi / 2 or angle > 3 * np.pi / 2:
            ha = "left"
        elif np.pi / 2 < angle < 3 * np.pi / 2:
            ha = "right"
        ax.text(angle, 1.18, label,
                ha=ha, va="center",
                transform=ax.get_xaxis_transform(),
                fontsize=17)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.5))

    # ── (a)(b) label — just outside top-left of polar frame ──────────────────
    if abc:
        ax.text(-0.05, 1.3, abc,
                transform=ax.transAxes,
                fontsize=17, fontweight="bold", va="top", ha="left")

#──────────────────────────────────────────────────────────────────────────────
#  PTX FLEXIBILITY METRICS FUNCTIONS
#──────────────────────────────────────────────────────────────────────────────
def responsiveness(flow: pd.Series, capacity: float, eps: float = 1e-6) -> float:
    """Fraction of online hours with active ramping."""
    if capacity <= 0:
        return 0.0
    threshold    = eps #* capacity
    online       = flow > threshold
    ramp         = flow.diff().abs() > threshold
    online_hours = online.sum()
    if online_hours == 0:
        return 0.0
    return float((online & ramp).sum() / online_hours)

def ptx_upward_flex(flow: pd.Series, capacity: float) -> float:
    """Average headroom / capacity — how much more the component could absorb."""
    if capacity <= 0:
        return 0.0
    return float((capacity - flow).clip(lower=0).mean() / capacity)

def ptx_downward_flex(flow: pd.Series, capacity: float,
                      min_load_fraction: float = 0.0) -> float:
    """Average footroom / capacity — how much the component could reduce."""
    if capacity <= 0:
        return 0.0
    min_load_abs = min_load_fraction * capacity
    online       = flow > min_load_abs
    if online.sum() == 0:
        return 0.0
    return float((flow - min_load_abs).clip(lower=0)[online].mean() / capacity)

def ramp_utilisation(flow: pd.Series, capacity: float, min_load_fraction: float = 0.0) -> float:
    """
    Mean absolute hourly ramp / capacity.
    Measures how intensely the component modulates, not just how often.
    Natural PtX equivalent of the storage load-shift index.
    Bounded [0, 1]: 1 would mean swinging full capacity every single hour.
    """
    if capacity <= 0:
        return 0.0
    min_load_abs      = min_load_fraction * capacity
    controllable_range = capacity - min_load_abs
    if controllable_range <= 0:
        return 0.0
    online_mask = flow > min_load_abs
    ramps_online = flow.diff().abs()[online_mask]
    if ramps_online.empty:
        return 0.0
    return float(ramps_online.mean() / controllable_range)




# =============================================================================
# 1. STORAGE-BASED FLEXIBILITY


def storage_utilisation_rate(soc: pd.Series, capacity: float) -> float:
    """
    Average fraction of storage capacity that is occupied.
    Analogous to a capacity factor for energy storage.
    
    Formula: mean(SoC[t]) / capacity
    Range  : 0-1  (1 = always full, 0 = always empty)

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
    maximum energy that **could** have been discharged during the same hours.

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


def storage_flexibility_band(
    soc: pd.Series,
    capacity: float,
    power_rating: float,
    min_soc_fraction: float,
    dt: float = 1.0,
) -> dict:
    """
    At each hour, computes:
      - upward flexibility   = remaining headroom (can absorb more)
      - downward flexibility = current SoC above minimum (can release this much)

    Both energy AND power constraints are applied at each timestep:
      - upward   = min(power_rating * dt,  capacity - soc[t])
      - downward = min(power_rating * dt,  soc[t] - min_soc_abs)

    Returns annual averages of both.

    Parameters
    ----------
    soc               : hourly state of charge [MWh]
    capacity          : storage energy capacity [MWh]
    power_rating      : maximum charge/discharge power [MW]
    min_soc_fraction  : minimum SoC as fraction of capacity (e.g. 0.2 for 20%)
    dt                : timestep duration [h], default 1

    Unit: MWh (average available flexibility at any given hour)
    """
    if capacity <= 0 or power_rating <= 0:
        return {"upward_mwh": 0.0, "downward_mwh": 0.0}

    min_soc_abs  = min_soc_fraction * capacity
    max_energy_dt = power_rating * dt             # max energy exchangeable per timestep

    upward   = np.minimum(max_energy_dt, capacity - soc)              # space to charge
    downward = np.minimum(max_energy_dt, (soc - min_soc_abs).clip(lower=0.0))  # energy to discharge

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




#──────────────────────────────────────────────────────────────────────────────
#  CURTAILMENT ANALYSIS
#──────────────────────────────────────────────────────────────────────────────


def add_curtailment_limit(model, bus_electricity, bus_H2, bus_heat,
                                electricity_curtailement, H2_curtailement, heat_curtailement,
                                pv, wind, fraction):
    """
    Constrains total annual curtailment across all carriers (electricity, H2, heat)
    to a fraction of total annual RES generation (PV + Wind).

    Parameters
    ----------
    model                    : solph.Model
    bus_electricity          : solph.Bus
    bus_H2                   : solph.Bus
    bus_heat                 : solph.Bus
    electricity_curtailement : solph.components.Sink
    H2_curtailement          : solph.components.Sink
    heat_curtailement        : solph.components.Sink
    pv                       : solph.components.Source
    wind                     : solph.components.Source
    fraction                 : float — e.g. 0.10 for 10%
    """
 
    total_curtailment = po.quicksum(
        model.flow[bus_electricity, electricity_curtailement, t] +
        model.flow[bus_H2,          H2_curtailement,          t] +
        model.flow[bus_heat,         heat_curtailement,        t]
        for t in model.TIMESTEPS
    )
    total_generation = po.quicksum(
        model.flow[pv,   bus_electricity, t] +
        model.flow[wind, bus_electricity, t]
        for t in model.TIMESTEPS
    )
    model.curtailment_limit_total = po.Constraint(
        expr=total_curtailment == fraction * total_generation
    )
    return model






