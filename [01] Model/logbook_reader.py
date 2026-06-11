'''Author: Gonçalo Costa Pina
Reads and prints all logbook entries in a human-readable format.
Run from [01] Model folder.
'''

import pandas as pd
from pathlib import Path
import sys

# ── Path ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent

# Accept logbook path as CLI argument, or fall back to a default
if len(sys.argv) > 1:
    logbook_path = Path(sys.argv[1])
else:
    logbook_path = DATA_DIR / "Logbooks" / "Logbook_NO_CAVERN.csv"

if not logbook_path.exists():
    print(f"No logbook found at: {logbook_path}")
    exit()

df = pd.read_csv(logbook_path)
print(f"Logbook loaded: {len(df)} run(s) found\n")


# ── Optional: override labels and units for known fields ─────────────────────
# Any field not listed here will be auto-labelled from its column name.
# Format: "group.variable": ("Human-readable label", "unit")
FIELD_MAP_OVERRIDES = {
    "parameters.wacc":                              ("WACC",                                    "-"),
    "levelized_cost.lcoh2_eur_per_mwh":             ("LCOH2",                                   "€/MWh_H2"),
    "levelized_cost.lcoh2_eur_per_kg":              ("LCOH2",                                   "€/kg_H2"),
    "financial.npv_final_eur":                      ("NPV (final)",                             "€"),
    "financial.irr":                                ("IRR",                                     "-"),
    "financial.payback_discounted_yr":              ("Payback time (discounted)",               "yr"),
    # Add more overrides here as needed — everything else is auto-labelled
}

# ── Unit inference from column name suffix ────────────────────────────────────
UNIT_SUFFIX_MAP = {
    "_eur_per_mwh": "€/MWh",
    "_eur_per_kg":  "€/kg",
    "_eur_per_yr":  "€/yr",
    "_eur":         "€",
    "_mwh":         "MWh",
    "_mw":          "MW",
    "_kg":          "kg",
    "_yr":          "yr",
    "_fraction":    "-",
    "_factor":      "-",
}

def infer_unit(col: str) -> str:
    """Infer a unit string from common column name suffixes."""
    col_lower = col.lower()
    for suffix, unit in UNIT_SUFFIX_MAP.items():
        if col_lower.endswith(suffix):
            return unit
    return "-"

def auto_label(col: str) -> str:
    """
    Convert a dot-separated column name to a readable label.
    e.g. "capacities.capacity_pv_mw" → "Capacity pv mw"
    """
    # Take only the variable part (after the last dot)
    var_part = col.split(".")[-1]
    return var_part.replace("_", " ").capitalize()

def get_label_unit(col: str):
    """Return (label, unit) for a column, using overrides where available."""
    if col in FIELD_MAP_OVERRIDES:
        return FIELD_MAP_OVERRIDES[col]
    return auto_label(col), infer_unit(col)


# ── Group columns by their prefix for section headers ────────────────────────
def get_group(col: str) -> str:
    """Return the group prefix of a column (everything before the first dot)."""
    return col.split(".")[0] if "." in col else "general"


# ── Format a single value ─────────────────────────────────────────────────────
def fmt(val, unit: str) -> str:
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float)):
        if unit in ("€", "€/yr", "MWh/yr", "kg/yr", "MWh", "kg"):
            return f"{val:>15,.0f}"
        elif unit in ("MW", "MWh", "€/MWh", "€/MWh_H2", "€/kg_H2", "€/kg"):
            return f"{val:>15,.2f}"
        else:
            return f"{val:>15.4f}"
    return f"{val:>15}"


# ── Collect all data columns (exclude 'scenario' metadata) ───────────────────
data_cols = [c for c in df.columns if c != "scenario"]

# ── Table layout ──────────────────────────────────────────────────────────────
run_headers = [
    f"Run {i+1} ({df.iloc[i].get('scenario', 'N/A')})"
    for i in range(len(df))
]

col_width_label = 50
col_width_val   = 18

divider_width = col_width_label + (col_width_val + 3) * len(run_headers) + 10

header_line = (
    " " * col_width_label + " | "
    + " | ".join(f"{h:>{col_width_val}}" for h in run_headers)
)
print("=" * divider_width)
print(header_line)
print("=" * divider_width)

# ── Print rows, inserting section headers when the group changes ──────────────
current_group = None

for col in data_cols:
    group = get_group(col)

    # Print section header when group changes
    if group != current_group:
        current_group = group
        section_title = group.replace("_", " ").upper()
        print(f"\n  [{section_title}]")

    label, unit = get_label_unit(col)
    row_vals = [fmt(run[col], unit) for _, run in df.iterrows()]
    line = (
        f"  {label:<{col_width_label - 2}} | "
        + " | ".join(row_vals)
        + f"   {unit}"
    )
    print(line)

print("\n" + "=" * divider_width)