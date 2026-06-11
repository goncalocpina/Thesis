"""
run_optimization.py
===================
Main execution script for the H2 energy system optimization model.

Usage
-----
    python run_optimization.py [--solver cbc] [--pv data/pv.csv]
                               [--wind data/wind.csv]
                               [--demand data/h2_demand.csv]
                               [--output output/]
                               [--mip-gap 0.01]
                               [--time-limit 3600]

If CSV paths are not provided, synthetic profiles are used automatically
(useful for testing and demonstration).

Output
------
* ``output/dispatch.csv``        – hourly dispatch timeseries
* ``output/capacities.json``     – optimal technology sizes
* ``output/costs.json``          – cost breakdown and LCOH2
* ``output/figures/``            – visualization plots (PNG)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve H2 energy system optimization model (Pyomo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver", default="cbc",
                        help="Pyomo-compatible solver (glpk, cbc, cplex, gurobi)")
    parser.add_argument("--pv", default=None,
                        help="Path to PV capacity-factor CSV")
    parser.add_argument("--wind", default=None,
                        help="Path to wind capacity-factor CSV")
    parser.add_argument("--demand", default=None,
                        help="Path to hourly H2 demand CSV (MWh/h)")
    parser.add_argument("--output", default="output",
                        help="Output directory for results and figures")
    parser.add_argument("--mip-gap", type=float, default=0.01,
                        help="Relative MIP optimality gap tolerance (e.g. 0.01 = 1%%)")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Solver wall-clock time limit in seconds")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="Skip sensitivity analysis (faster run)")
    parser.add_argument("--sensitivity-points", type=int, default=4,
                        help="Number of grid points for sensitivity sweeps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 0. Imports (deferred so --help works without Pyomo installed)
    # ------------------------------------------------------------------
    from data_loader import load_profiles
    from pyomo_model import build_model, solve_model
    from results_extractor import extract_all, validate_energy_balance, save_results
    from visualizations import save_all_plots

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("H2 Energy System Optimization Model")
    print("=" * 60)
    print("\n[1] Loading input profiles …")
    profiles = load_profiles(
        pv_path=args.pv,
        wind_path=args.wind,
        demand_path=args.demand,
    )
    print(f"    PV profile    : {profiles['pv'].shape[0]} hours, "
          f"mean CF = {profiles['pv'].mean():.3f}")
    print(f"    Wind profile  : {profiles['wind'].shape[0]} hours, "
          f"mean CF = {profiles['wind'].mean():.3f}")
    print(f"    H2 demand     : {profiles['demand'].shape[0]} hours, "
          f"total = {profiles['demand'].sum():.0f} MWh")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    print("\n[2] Building Pyomo optimization model …")
    model = build_model(
        pv_cf=profiles["pv"],
        wind_cf=profiles["wind"],
        h2_demand=profiles["demand"],
    )
    print("    Model built successfully.")

    # ------------------------------------------------------------------
    # 3. Solve
    # ------------------------------------------------------------------
    print(f"\n[3] Solving with '{args.solver}' …")
    print(f"    MIP gap tolerance : {args.mip_gap * 100:.1f}%")
    print(f"    Time limit        : {args.time_limit}s")
    model, status = solve_model(
        model,
        solver_name=args.solver,
        tee=True,
        mip_gap=args.mip_gap,
        time_limit_seconds=args.time_limit,
    )

    if not status["optimal"]:
        print("\n[ERROR] Optimization did not converge to an optimal solution.")
        print(f"        Termination condition: {status['termination']}")
        sys.exit(1)
    print("    Optimal solution found.")

    # ------------------------------------------------------------------
    # 4. Extract results
    # ------------------------------------------------------------------
    print("\n[4] Extracting results …")
    results = extract_all(model)

    caps = results["capacities"]
    costs = results["costs"]

    print("\n  -- Optimal Capacities --")
    print(f"    PV              : {caps['pv_mw']:.2f} MW")
    print(f"    Wind            : {caps['wind_mw']:.2f} MW")
    print(f"    Electrolyser    : {caps['electrolyser_mw']:.2f} MW")
    print(f"    Battery (power) : {caps['battery_power_mw']:.2f} MW")
    print(f"    Battery (energy): {caps['battery_energy_mwh']:.2f} MWh")
    print(f"    Salt cavern     : {caps['cavern_mwh']:.2f} MWh_H2")

    print("\n  -- Annual Costs --")
    print(f"    PV              : {(costs['pv_capex_annual'] + costs['pv_opex_annual'])/1e3:.1f} k€/yr")
    print(f"    Wind            : {(costs['wind_capex_annual'] + costs['wind_opex_annual'])/1e3:.1f} k€/yr")
    print(f"    Electrolyser    : {(costs['elec_capex_annual'] + costs['elec_opex_annual'])/1e3:.1f} k€/yr")
    print(f"    Battery         : {(costs['batt_capex_annual'] + costs['batt_opex_annual'])/1e3:.1f} k€/yr")
    print(f"    Salt cavern     : {(costs['cavern_capex_annual'] + costs['cavern_opex_annual'])/1e3:.1f} k€/yr")
    print(f"    TOTAL           : {costs['total_annual_cost']/1e6:.3f} M€/yr")
    print(f"    LCOH2           : {costs['lcoh2']:.2f} €/MWh_H2")

    # ------------------------------------------------------------------
    # 5. Validate energy balance
    # ------------------------------------------------------------------
    print("\n[5] Validating energy balance …")
    balance_ok = validate_energy_balance(results)
    if not balance_ok:
        print("    [WARNING] Energy balance check failed – review results carefully.")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    print(f"\n[6] Saving results to '{args.output}/' …")
    save_results(results, output_dir=args.output)

    # ------------------------------------------------------------------
    # 7. Generate visualizations
    # ------------------------------------------------------------------
    print(f"\n[7] Generating visualizations …")
    fig_dir = Path(args.output) / "figures"
    save_all_plots(results, output_dir=fig_dir)

    # ------------------------------------------------------------------
    # 8. Sensitivity analysis (optional)
    # ------------------------------------------------------------------
    if not args.no_sensitivity:
        print(f"\n[8] Running sensitivity analysis ({args.sensitivity_points} points per axis) …")
        from sensitivity_analysis import run_all_sweeps
        run_all_sweeps(
            solver=args.solver,
            n_points=args.sensitivity_points,
            output_dir=Path(args.output) / "sensitivity",
            mip_gap=args.mip_gap,
            time_limit_seconds=args.time_limit,
        )
    else:
        print("\n[8] Sensitivity analysis skipped (--no-sensitivity).")

    print("\n" + "=" * 60)
    print("DONE.  Results saved to:", Path(args.output).resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()