import pyomo.environ as pe
import pyomo.opt as po
from tabulate import tabulate
import numpy as np

# Time periods
T = 24
time_periods = np.arange(1, T+1)

# Create the model
model = pe.ConcreteModel()

# Indexed sets
model.t = pe.Set(initialize=time_periods)

# Decision Variables
model.P_ELEC = pe.Var(model.t, domain=pe.NonNegativeReals)
model.P_FC = pe.Var(model.t, domain=pe.NonNegativeReals)
model.mH2_CHP = pe.Var(model.t, domain=pe.NonNegativeReals)
model.B_ELEC = pe.Var(model.t, domain=pe.Binary)
model.B_FC = pe.Var(model.t, domain=pe.Binary)
model.mH2_plus = pe.Var(domain=pe.NonNegativeReals)
model.mH2_minus = pe.Var(domain=pe.NonNegativeReals)

# Parameters
model.pi_H2_CHP = pe.Param(initialize=0.03)  # price CHP pays per unit of H2 to the H2PP operator
model.mH2_SP = pe.Param(initialize=5)  # set point of H2 mass in tank
model.P_min_ELEC = pe.Param(initialize=0)  # minimum operational limit of electrolyser
model.P_max_ELEC = pe.Param(initialize=5)  # maximum operational limit of electrolyser
model.P_min_FC = pe.Param(initialize=0)  # minimum operational limit of fuel cell
model.P_max_FC = pe.Param(initialize=5)  # maximum operational limit of fuel cell
model.eta_ELEC = pe.Param(initialize=0.6)  # efficiency of electrolyser
model.eta_FC = pe.Param(initialize=0.6)  # fuel cell efficiency
model.HHV_H2 = pe.Param(initialize=39.4)  # higher heating value (HHV) of H2 (39.4 kWh/kg)


# Initialize these with sinusoidal time-dependent data
def pi_e_B_init(model, t):
    return 0.05 + 0.02 * np.sin(2 * np.pi * (t-1) / 24)

model.pi_e_B = pe.Param(model.t, initialize=pi_e_B_init, mutable=True)  # price at which the H2PP buys electricity from the market

def pi_e_S_init(model, t):
    return 0.04 + 0.01 * np.sin(2 * np.pi * (t-1) / 24)

model.pi_e_S = pe.Param(model.t, initialize=pi_e_S_init, mutable=True)  # price at which the fuel cell sells its electricity in the market

model.mH2_req = pe.Param(model.t, initialize=3, mutable=True)  # maximum H2 required by the CHP

# Constraints

# Electrolyser constraints
def elec_H2_prod(model, t):
    return model.P_ELEC[t] * model.eta_ELEC / model.HHV_H2 == model.mH2_CHP[t]

model.elec_H2_prod = pe.Constraint(model.t, rule=elec_H2_prod)

def elec_min(model, t):
    return model.P_ELEC[t] >= model.P_min_ELEC * model.B_ELEC[t]

model.elec_min = pe.Constraint(model.t, rule=elec_min)

def elec_max(model, t):
    return model.P_ELEC[t] <= model.P_max_ELEC * model.B_ELEC[t]

model.elec_max = pe.Constraint(model.t, rule=elec_max)

# Fuel cell constraints
def fuel_H2_cons(model, t):
    return model.P_FC[t] == model.mH2_CHP[t] * model.eta_FC * model.HHV_H2

model.fuel_H2_cons = pe.Constraint(model.t, rule=fuel_H2_cons)

def fuel_min(model, t):
    return model.P_FC[t] >= model.P_min_FC * model.B_FC[t]

model.fuel_min = pe.Constraint(model.t, rule=fuel_min)

def fuel_max(model, t):
    return model.P_FC[t] <= model.P_max_FC * model.B_FC[t]

model.fuel_max = pe.Constraint(model.t, rule=fuel_max)

# Constraint on electrolyser and fuel cell
def elec_fuel_operational(model, t):
    return model.B_ELEC[t] + model.B_FC[t] <= 1

model.elec_fuel_operational = pe.Constraint(model.t, rule=elec_fuel_operational)

# CHP microturbine constraint
def chp_demand(model, t):
    return model.mH2_CHP[t] <= model.mH2_req[t]

model.chp_demand = pe.Constraint(model.t, rule=chp_demand)

# H2 storage constraint at the end of the day
# model.tank_balance = Constraint(expr=model.mH2_plus - model.mH2_minus == model.mH2_plus - model.mH2_minus)

def tank_balance_rule(model):
    return sum(model.mH2_CHP[t] for t in model.t) == model.mH2_plus - model.mH2_minus

model.tank_balance = pe.Constraint(rule=tank_balance_rule)


# Average electricity prices
avg_pi_e_B_td = sum(model.pi_e_B[t] for t in model.t) / len(model.t)
avg_pi_e_B_tmw = sum(model.pi_e_B[t+1] for t in model.t if t < T) / (len(model.t) - 1)

# Penalty factors
model.pi_H2_plus = pe.Param(initialize=avg_pi_e_B_td - avg_pi_e_B_tmw)  # Penalty factor for being above the set point
model.pi_H2_minus = pe.Param(initialize=avg_pi_e_B_td - avg_pi_e_B_tmw)  # Penalty factor for being below the set point


# Objective Function
def objective_rule(model):
    return (
        sum(
            -model.P_ELEC[t] * model.pi_e_B[t] + model.P_FC[t] * model.pi_e_S[t] + model.mH2_CHP[t] * model.pi_H2_CHP
            for t in model.t
        )
        - model.mH2_plus * model.pi_H2_plus
        + model.mH2_minus * model.pi_H2_minus
    )

model.objective = pe.Objective(rule=objective_rule, sense=pe.maximize)


# Solve the model
solver = po.SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# Check solver status and termination condition
if results.solver.status == pe.SolverStatus.ok and results.solver.termination_condition == pe.TerminationCondition.optimal:
    print("Solver Status: Optimal solution found")
    print("Termination Condition:", results.solver.termination_condition)

    # Create a table of results
    table = []

    # Add header row
    header = ["Time", "P_ELEC", "P_FC", "mH2_CHP"]
    table.append(header)

    # Add rows for each time period
    for t in model.t:
        row = [t, "{:.10f}".format(model.P_ELEC[t].value), "{:.10f}".format(model.P_FC[t].value), "{:.10f}".format(model.mH2_CHP[t].value)]
        table.append(row)

    # Add row for objective function value
    objective_row = ["Objective", model.objective()]
    table.append(objective_row)

    # Print the table
    print(tabulate(table, headers="firstrow"))

    # Print additional results
    print("Penalty factor for being above the set point:")
    print("pi_H2_plus = {:.10f}".format(model.pi_H2_plus.value))
    print("Penalty factor for being below the set point:")
    print("pi_H2_minus = {:.10f}".format(model.pi_H2_minus.value))
    print("Amount of H2 in the tank above the setpoint:")
    print("mH2_plus = {:.10f}".format(model.mH2_plus.value))
    print("Amount of H2 in the tank below the setpoint:")
    print("mH2_minus = {:.10f}".format(model.mH2_minus.value))
elif results.solver.termination_condition == pe.TerminationCondition.infeasible:
    print("Solver Status: Problem is infeasible")
elif results.solver.termination_condition == pe.TerminationCondition.unbounded:
    print("Solver Status: Problem is unbounded")
elif results.solver.termination_condition == pe.TerminationCondition.infeasibleOrUnbounded:
    print("Solver Status: Problem is infeasible or unbounded")
else:
    print("Solver Status:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)
