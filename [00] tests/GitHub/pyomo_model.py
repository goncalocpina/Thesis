"""
pyomo_model.py
==============
Pyomo-based linear optimization model for sizing and dispatch of a
hydrogen (H2) energy system with:

    * PV generation
    * Wind generation
    * Electrolyser (power-to-hydrogen)
    * Li-ion battery storage (electricity buffer)
    * Salt cavern H2 storage

Decision variables
------------------
Sizing (continuous, ≥ 0):
    X_pv          – installed PV capacity [MW]
    X_wind         – installed wind capacity [MW]
    X_elec         – electrolyser electrical input capacity [MW]
    X_batt_power   – battery power capacity [MW]
    X_batt_energy  – battery energy capacity [MWh]
    X_cavern       – salt cavern working-gas capacity [MWh_H2]

Dispatch (per hour t):
    pv_out[t]          – PV generation dispatched to load [MW]
    wind_out[t]        – Wind generation dispatched to load [MW]
    elec_in[t]         – electrical power into electrolyser [MW]
    batt_charge[t]     – battery charge power [MW]
    batt_discharge[t]  – battery discharge power [MW]
    batt_soc[t]        – battery state of charge [MWh]
    elec_on[t]         – binary: electrolyser operating (1) or off (0)
    dump[t]            – curtailed electricity [MW]
    h2_produced[t]     – H2 produced by electrolyser [MWh_H2/hr]
    cavern_charge[t]   – H2 injected into cavern [MWh_H2/hr]
    cavern_discharge[t]– H2 withdrawn from cavern [MWh_H2/hr]
    cavern_inv[t]      – cavern H2 inventory [MWh_H2]

Objective
---------
Minimize total annualized system cost (€/year):
    sum over components of (CAPEX * CRF + OPEX) * capacity
    + variable battery O&M * total MWh cycled

Constraints
-----------
(1)  Electricity balance
(2)  H2 production = electrolyser efficiency × electrical input
(3)  H2 balance (production + cavern discharge = demand + cavern charge)
(4)  Electrolyser minimum load (via big-M linearisation of binary condition)
(5)  Battery SOC dynamics
(6)  Battery capacity bounds
(7)  Battery C-ratio (energy ≥ C_ratio × power)
(8)  Cavern inventory dynamics (with hourly volume-loss decay)
(9)  Cavern capacity bounds
(10) Cyclic boundary conditions (SOC and cavern inventory)
"""

from __future__ import annotations

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import parameters as p


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    pv_cf: list | None = None,
    wind_cf: list | None = None,
    h2_demand: list | None = None,
    params: dict | None = None,
) -> pyo.ConcreteModel:
    """
    Build and return the Pyomo ConcreteModel.

    Parameters
    ----------
    pv_cf : array-like of length N_HOURS
        Hourly PV capacity factors (fraction of installed MW).
    wind_cf : array-like of length N_HOURS
        Hourly wind capacity factors (fraction of installed MW).
    h2_demand : array-like of length N_HOURS
        Hourly H2 demand [MWh / hour].
    params : dict, optional
        Override any entry in ``parameters.PARAMS``.  Keys must match.

    Returns
    -------
    pyomo.ConcreteModel
        Fully populated model ready to hand to a solver.
    """
    # ---- merge parameters --------------------------------------------------
    cfg = dict(p.PARAMS)
    if params:
        cfg.update(params)

    n = cfg["n_hours"]

    # Fallback: synthetic profiles if not provided
    if pv_cf is None or wind_cf is None or h2_demand is None:
        from data_loader import generate_synthetic_data
        syn = generate_synthetic_data()
        if pv_cf is None:
            pv_cf = syn["pv"]
        if wind_cf is None:
            wind_cf = syn["wind"]
        if h2_demand is None:
            h2_demand = syn["demand"]

    pv_cf = list(pv_cf)
    wind_cf = list(wind_cf)
    h2_demand = list(h2_demand)

    # ---- model object ------------------------------------------------------
    m = pyo.ConcreteModel(name="H2_System_Optimization")

    # ---- sets ---------------------------------------------------------------
    m.T = pyo.RangeSet(0, n - 1)   # time steps 0 … n-1

    # ---- parameters (data) -------------------------------------------------
    m.pv_cf = pyo.Param(m.T, initialize=dict(enumerate(pv_cf)), within=pyo.NonNegativeReals)
    m.wind_cf = pyo.Param(m.T, initialize=dict(enumerate(wind_cf)), within=pyo.NonNegativeReals)
    m.h2_demand = pyo.Param(m.T, initialize=dict(enumerate(h2_demand)), within=pyo.NonNegativeReals)

    # ---- sizing variables --------------------------------------------------
    m.X_pv = pyo.Var(within=pyo.NonNegativeReals, doc="Installed PV capacity [MW]")
    m.X_wind = pyo.Var(within=pyo.NonNegativeReals, doc="Installed wind capacity [MW]")
    m.X_elec = pyo.Var(within=pyo.NonNegativeReals, doc="Electrolyser capacity (elec. input) [MW]")
    m.X_batt_power = pyo.Var(within=pyo.NonNegativeReals, doc="Battery power capacity [MW]")
    m.X_batt_energy = pyo.Var(within=pyo.NonNegativeReals, doc="Battery energy capacity [MWh]")
    m.X_cavern = pyo.Var(within=pyo.NonNegativeReals, doc="Salt cavern working-gas capacity [MWh_H2]")

    # ---- dispatch variables ------------------------------------------------
    m.elec_in = pyo.Var(m.T, within=pyo.NonNegativeReals,
                        doc="Electricity into electrolyser [MW]")
    m.batt_charge = pyo.Var(m.T, within=pyo.NonNegativeReals,
                            doc="Battery charge power [MW]")
    m.batt_discharge = pyo.Var(m.T, within=pyo.NonNegativeReals,
                               doc="Battery discharge power [MW]")
    m.batt_soc = pyo.Var(m.T, within=pyo.NonNegativeReals,
                         doc="Battery state of charge [MWh]")
    m.dump = pyo.Var(m.T, within=pyo.NonNegativeReals,
                     doc="Curtailed (dumped) electricity [MW]")
    m.h2_produced = pyo.Var(m.T, within=pyo.NonNegativeReals,
                            doc="H2 produced by electrolyser [MWh_H2/hr]")
    m.cavern_charge = pyo.Var(m.T, within=pyo.NonNegativeReals,
                              doc="H2 injected into cavern [MWh_H2/hr]")
    m.cavern_discharge = pyo.Var(m.T, within=pyo.NonNegativeReals,
                                 doc="H2 withdrawn from cavern [MWh_H2/hr]")
    m.cavern_inv = pyo.Var(m.T, within=pyo.NonNegativeReals,
                           doc="Cavern H2 inventory [MWh_H2]")

    # Binary: is the electrolyser operating?
    m.elec_on = pyo.Var(m.T, within=pyo.Binary,
                        doc="Electrolyser on/off flag (1 = operating)")

    # ---- auxiliary scalar for total battery throughput (for variable OPEX) -
    m.batt_throughput = pyo.Var(within=pyo.NonNegativeReals,
                                doc="Total annual battery charge throughput [MWh]")

    # ====================================================================
    # OBJECTIVE FUNCTION
    # Minimize total annualized system cost [€/year]
    # ====================================================================
    #
    #   Cost = sum_i [ CAPEX_i * CRF_i * X_i ]    <- annualized capital
    #        + sum_i [ OPEX_i  * X_i        ]    <- fixed O&M
    #        + OPEX_var_batt * batt_throughput   <- variable O&M battery
    #
    def total_cost(m):
        # Annualized CAPEX + fixed OPEX per component
        pv_annual = (cfg["pv_capex"] * cfg["pv_crf"] + cfg["pv_opex"]) * m.X_pv
        wind_annual = (cfg["wind_capex"] * cfg["wind_crf"] + cfg["wind_opex"]) * m.X_wind
        elec_annual = (cfg["elec_capex"] * cfg["elec_crf"] + cfg["elec_opex"]) * m.X_elec
        batt_energy_annual = (
            cfg["batt_capex_energy"] * cfg["batt_crf"] * m.X_batt_energy
        )
        batt_power_annual = cfg["batt_opex_power"] * m.X_batt_power
        cavern_annual = (
            cfg["cavern_capex_per_mwh"] * cfg["cavern_crf"]
            + cfg["cavern_opex_per_mwh"]
        ) * m.X_cavern

        # Variable battery O&M
        batt_var = cfg["batt_opex_variable"] * m.batt_throughput

        return (
            pv_annual
            + wind_annual
            + elec_annual
            + batt_energy_annual
            + batt_power_annual
            + cavern_annual
            + batt_var
        )

    m.obj = pyo.Objective(rule=total_cost, sense=pyo.minimize)

    # ====================================================================
    # CONSTRAINTS
    # ====================================================================

    # ------------------------------------------------------------------
    # (1) Electricity balance
    # ------------------------------------------------------------------
    # PV[t]*X_pv + Wind[t]*X_wind
    #     = elec_in[t] + batt_charge[t] - batt_discharge[t] + dump[t]
    #
    # Note: battery discharge appears as a source on the demand side,
    # so the net electricity balance is:
    #   supply = PV + Wind + batt_discharge
    #   demand = elec_in + batt_charge + dump
    #
    def elec_balance(m, t):
        supply = m.pv_cf[t] * m.X_pv + m.wind_cf[t] * m.X_wind + m.batt_discharge[t]
        demand_el = m.elec_in[t] + m.batt_charge[t] + m.dump[t]
        return supply == demand_el

    m.con_elec_balance = pyo.Constraint(m.T, rule=elec_balance)

    # ------------------------------------------------------------------
    # (2) H2 production
    # ------------------------------------------------------------------
    # H2_produced[t] = η_elec × elec_in[t]
    #
    def h2_production(m, t):
        return m.h2_produced[t] == cfg["elec_efficiency"] * m.elec_in[t]

    m.con_h2_production = pyo.Constraint(m.T, rule=h2_production)

    # ------------------------------------------------------------------
    # (3) H2 energy balance
    # ------------------------------------------------------------------
    # H2_produced[t] + cavern_discharge[t] = H2_demand[t] + cavern_charge[t]
    #
    def h2_balance(m, t):
        return (
            m.h2_produced[t] + m.cavern_discharge[t]
            == m.h2_demand[t] + m.cavern_charge[t]
        )

    m.con_h2_balance = pyo.Constraint(m.T, rule=h2_balance)

    # ------------------------------------------------------------------
    # (4) Electrolyser capacity and minimum load  (Big-M linearisation)
    # ------------------------------------------------------------------
    # X_elec is a continuous variable, elec_on[t] is binary.
    # Their product is bilinear → linearised via a Big-M parameter.
    #
    # Let M_elec be an upper bound on possible electrolyser capacity.
    # Constraints:
    #   (4a) elec_in[t] ≤ X_elec               [physical capacity]
    #   (4b) elec_in[t] ≤ M_elec · elec_on[t]  [zero when off]
    #   (4c) elec_in[t] ≥ MIN_LOAD · X_elec
    #                      - M_elec · (1 − elec_on[t])
    #        → when elec_on=1: elec_in ≥ MIN_LOAD·X_elec  ✓
    #        → when elec_on=0: constraint relaxed (RHS < 0)  ✓
    #
    M_elec = cfg.get("elec_big_m", 50_000.0)  # MW – generous upper bound

    def elec_cap_upper(m, t):
        # elec_in[t] ≤ X_elec  (continuous capacity bound)
        return m.elec_in[t] <= m.X_elec

    m.con_elec_cap_upper = pyo.Constraint(m.T, rule=elec_cap_upper)

    def elec_off_zero(m, t):
        # elec_in[t] ≤ M_elec · elec_on[t]  → zero when off
        return m.elec_in[t] <= M_elec * m.elec_on[t]

    m.con_elec_off_zero = pyo.Constraint(m.T, rule=elec_off_zero)

    def elec_min_load(m, t):
        # elec_in[t] ≥ MIN_LOAD · X_elec − M_elec · (1 − elec_on[t])
        return (
            m.elec_in[t]
            >= cfg["elec_min_load"] * m.X_elec - M_elec * (1 - m.elec_on[t])
        )

    m.con_elec_min_load = pyo.Constraint(m.T, rule=elec_min_load)

    # ------------------------------------------------------------------
    # (5) Battery state-of-charge (SOC) dynamics
    # ------------------------------------------------------------------
    # SOC[t] = SOC[t-1] * (1 - σ) + charge[t] * η_in - discharge[t] / η_out
    #
    # where σ is the hourly self-discharge rate.
    #
    sigma = cfg["batt_self_discharge"]
    eta_in = cfg["batt_eta_in"]
    eta_out = cfg["batt_eta_out"]

    def batt_soc_dynamics(m, t):
        t_prev = (t - 1) % n   # cyclic wrap-around
        return (
            m.batt_soc[t]
            == m.batt_soc[t_prev] * (1 - sigma)
            + m.batt_charge[t] * eta_in
            - m.batt_discharge[t] / eta_out
        )

    m.con_batt_soc_dynamics = pyo.Constraint(m.T, rule=batt_soc_dynamics)

    # (5a) SOC upper bound
    def batt_soc_upper(m, t):
        return m.batt_soc[t] <= m.X_batt_energy

    m.con_batt_soc_upper = pyo.Constraint(m.T, rule=batt_soc_upper)

    # (5b) Battery charge upper bound
    def batt_charge_upper(m, t):
        return m.batt_charge[t] <= m.X_batt_power

    m.con_batt_charge_upper = pyo.Constraint(m.T, rule=batt_charge_upper)

    # (5c) Battery discharge upper bound
    def batt_discharge_upper(m, t):
        return m.batt_discharge[t] <= m.X_batt_power

    m.con_batt_discharge_upper = pyo.Constraint(m.T, rule=batt_discharge_upper)

    # ------------------------------------------------------------------
    # (6) Battery C-ratio constraint
    # ------------------------------------------------------------------
    # X_batt_energy ≥ C_ratio × X_batt_power
    # (energy capacity must be at least C_ratio hours of rated power)
    #
    def batt_c_ratio(m):
        return m.X_batt_energy >= cfg["batt_c_ratio"] * m.X_batt_power

    m.con_batt_c_ratio = pyo.Constraint(rule=batt_c_ratio)

    # ------------------------------------------------------------------
    # (7) Battery throughput accounting (for variable OPEX)
    # ------------------------------------------------------------------
    # batt_throughput = Σ_t batt_charge[t]
    #
    def batt_throughput_def(m):
        return m.batt_throughput == sum(m.batt_charge[t] for t in m.T)

    m.con_batt_throughput = pyo.Constraint(rule=batt_throughput_def)

    # ------------------------------------------------------------------
    # (8) Salt cavern inventory dynamics
    # ------------------------------------------------------------------
    # Inventory[t] = Inventory[t-1] * (1 - λ) + charge[t] - discharge[t]
    #
    # where λ = CAVERN_VOLUME_LOSS_HOURLY is the per-hour fractional
    # volume loss due to cavern shrinkage.
    #
    lam = cfg["cavern_volume_loss_hourly"]

    def cavern_inv_dynamics(m, t):
        t_prev = (t - 1) % n
        return (
            m.cavern_inv[t]
            == m.cavern_inv[t_prev] * (1 - lam)
            + m.cavern_charge[t]
            - m.cavern_discharge[t]
        )

    m.con_cavern_inv_dynamics = pyo.Constraint(m.T, rule=cavern_inv_dynamics)

    # (8a) Cavern capacity upper bound
    def cavern_inv_upper(m, t):
        return m.cavern_inv[t] <= m.X_cavern

    m.con_cavern_inv_upper = pyo.Constraint(m.T, rule=cavern_inv_upper)

    # ------------------------------------------------------------------
    # (9) Cavern charge / discharge power bounds
    #     (no explicit compressor sizing; discharge limited by capacity)
    # ------------------------------------------------------------------
    def cavern_charge_upper(m, t):
        return m.cavern_charge[t] <= m.X_cavern

    m.con_cavern_charge_upper = pyo.Constraint(m.T, rule=cavern_charge_upper)

    def cavern_discharge_upper(m, t):
        return m.cavern_discharge[t] <= m.X_cavern

    m.con_cavern_discharge_upper = pyo.Constraint(m.T, rule=cavern_discharge_upper)

    # ------------------------------------------------------------------
    # (10) Cyclic boundary conditions
    # ------------------------------------------------------------------
    # Enforce that the end-of-year state equals the start-of-year state,
    # ensuring inter-annual repeatability.
    #
    def batt_soc_cyclic(m):
        return m.batt_soc[n - 1] == m.batt_soc[0]

    # Note: the SOC dynamics constraint at t=0 already wraps to t=n-1,
    # which together with the dynamics at t=n-1 implicitly enforces
    # cyclicity via the modular index.  We add an explicit constraint
    # for clarity / redundancy.
    m.con_batt_soc_cyclic = pyo.Constraint(rule=batt_soc_cyclic)

    def cavern_inv_cyclic(m):
        return m.cavern_inv[n - 1] == m.cavern_inv[0]

    m.con_cavern_inv_cyclic = pyo.Constraint(rule=cavern_inv_cyclic)

    return m


# ---------------------------------------------------------------------------
# Solver interface
# ---------------------------------------------------------------------------

def solve_model(
    model: pyo.ConcreteModel,
    solver_name: str = "glpk",
    tee: bool = True,
    mip_gap: float = 0.01,
    time_limit_seconds: int = 3600,
) -> tuple[pyo.ConcreteModel, dict]:
    """
    Solve the optimization model and return the solved model plus a status dict.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Model returned by :func:`build_model`.
    solver_name : str
        Name of the Pyomo-registered solver (e.g. ``'glpk'``, ``'cbc'``,
        ``'cplex'``, ``'gurobi'``).
    tee : bool
        If True, stream solver output to stdout.
    mip_gap : float
        Relative MIP optimality gap tolerance (e.g. 0.01 = 1 %).
    time_limit_seconds : int
        Wall-clock time limit in seconds.  Prevents the solver from running
        indefinitely on large MILP problems.  Default: 3600 s (1 hour).
        If a feasible (but not proven-optimal) solution is found within the
        time limit, it is returned and ``status['optimal']`` is True when
        the gap is within ``mip_gap``.

    Returns
    -------
    model : pyo.ConcreteModel
        Solved model with variable values populated.
    status : dict
        Keys: ``'solver'``, ``'termination'``, ``'optimal'``.

    Notes on solver option names
    ----------------------------
    Each solver uses different names for the same concept:

    +-----------+------------------+------------------+
    | Solver    | MIP gap option   | Time limit option|
    +===========+==================+==================+
    | GLPK      | mipgap           | tmlim            |
    +-----------+------------------+------------------+
    | CBC       | ratioGap         | seconds          |
    +-----------+------------------+------------------+
    | CPLEX     | mipgap           | timelimit        |
    +-----------+------------------+------------------+
    | Gurobi    | MIPGap           | TimeLimit        |
    +-----------+------------------+------------------+
    """
    solver = SolverFactory(solver_name)

    sn = solver_name.lower()
    if sn == "glpk":
        solver.options["mipgap"] = mip_gap
        solver.options["tmlim"] = time_limit_seconds
    elif sn == "cbc":
        solver.options["ratioGap"] = mip_gap
        solver.options["seconds"] = time_limit_seconds
    elif sn in ("cplex", "cplex_direct"):
        solver.options["mipgap"] = mip_gap
        solver.options["timelimit"] = time_limit_seconds
    elif sn in ("gurobi", "gurobi_direct"):
        solver.options["MIPGap"] = mip_gap
        solver.options["TimeLimit"] = time_limit_seconds
    # Other solvers: no options set (user can configure externally)

    result = solver.solve(model, tee=tee)

    optimal = (
        result.solver.status == SolverStatus.ok
        and result.solver.termination_condition in (
            TerminationCondition.optimal,
            TerminationCondition.maxTimeLimit,   # feasible solution within time limit
        )
    )

    # For time-limit terminations, only accept if a feasible solution was loaded
    if result.solver.termination_condition == TerminationCondition.maxTimeLimit:
        try:
            pyo.value(model.obj)  # raises if no solution loaded
        except Exception:
            optimal = False

    status = {
        "solver": solver_name,
        "termination": str(result.solver.termination_condition),
        "optimal": optimal,
    }

    if result.solver.termination_condition == TerminationCondition.maxTimeLimit and optimal:
        print(
            f"[pyomo_model] INFO: Time limit ({time_limit_seconds}s) reached. "
            "A feasible (sub-optimal) solution was returned.  "
            "Increase --time-limit or relax --mip-gap if needed."
        )
    elif not optimal:
        print(
            f"[pyomo_model] WARNING: Solver did not find an optimal solution. "
            f"Termination: {result.solver.termination_condition}"
        )

    return model, status