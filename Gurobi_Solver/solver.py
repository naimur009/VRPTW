import math
import os
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# Pure Gurobi Baseline for Solomon-style VRPTW
# Full graph, no pruning, no GNN scores, no warm start
# ============================================================

# -----------------------
# Config
# -----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "nodes_0.csv")   # instance file

Q = 200                 # vehicle capacity
MAX_VEHICLES = 30       # upper bound on number of vehicles
TIME_LIMIT = 300        # seconds
MIP_GAP = 0.01
LAMBDA = 10000.0        # strong route penalty to prioritize fewer vehicles


# -----------------------
# Read instance
# -----------------------
df = pd.read_csv(CSV_PATH)

# Keep original Solomon ordering:
# row 0 = depot, rows 1..n-1 = customers
xcoord = df["x"].tolist()
ycoord = df["y"].tolist()
demand = df["demand"].tolist()
ready = df["ready"].tolist()
due = df["due"].tolist()
service = df["service"].tolist()

n_total = len(df)
N = list(range(n_total))
depot = 0
V = [i for i in N if i != depot]


# -----------------------
# Distance / travel time
# -----------------------
def euclid(i, j):
    return math.hypot(xcoord[i] - xcoord[j], ycoord[i] - ycoord[j])


c = {(i, j): euclid(i, j) for i in N for j in N if i != j}
t = c.copy()   # common assumption: travel time = Euclidean distance


# -----------------------
# Safe Big-M for time propagation
# For:
# tau[j] >= tau[i] + service[i] + t[i,j] - M[i,j]*(1-x[i,j])
# -----------------------
M = {}
for i, j in c.keys():
    M[i, j] = max(0.0, due[i] + service[i] + t[i, j] - ready[j])


# -----------------------
# Build model
# -----------------------
model = gp.Model("Pure_Gurobi_VRPTW_Baseline")

# Variables
x = model.addVars(c.keys(), vtype=GRB.BINARY, name="x")
tau = model.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="tau")
u = model.addVars(N, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="u")

# Objective:
# First reduce number of routes strongly, then reduce distance
model.setObjective(
    LAMBDA * gp.quicksum(x[depot, j] for j in V) +
    gp.quicksum(c[i, j] * x[i, j] for (i, j) in c.keys()),
    GRB.MINIMIZE
)


# ============================================================
# Constraints
# ============================================================

# (1) Each customer visited exactly once
for i in V:
    model.addConstr(
        gp.quicksum(x[i, j] for j in N if j != i) == 1,
        name=f"out_{i}"
    )
    model.addConstr(
        gp.quicksum(x[j, i] for j in N if j != i) == 1,
        name=f"in_{i}"
    )

# (2) Depot balance and vehicle limit
model.addConstr(
    gp.quicksum(x[depot, j] for j in V) ==
    gp.quicksum(x[i, depot] for i in V),
    name="depot_balance"
)

model.addConstr(
    gp.quicksum(x[depot, j] for j in V) <= MAX_VEHICLES,
    name="max_vehicles"
)

# (3) Time windows
for i in N:
    model.addConstr(tau[i] >= ready[i], name=f"ready_{i}")
    model.addConstr(tau[i] <= due[i], name=f"due_{i}")

# Fix depot start time
model.addConstr(tau[depot] == ready[depot], name="depot_time_fix")

# (4) Time propagation
# Usually we only need this when destination j is a customer.
for (i, j) in c.keys():
    if j != depot:
        model.addConstr(
            tau[j] >= tau[i] + service[i] + t[i, j] - M[i, j] * (1 - x[i, j]),
            name=f"time_{i}_{j}"
        )

# (5) Capacity / MTZ-style load propagation
model.addConstr(u[depot] == 0.0, name="u_depot")

for i in V:
    model.addConstr(u[i] >= demand[i], name=f"u_lb_{i}")
    model.addConstr(u[i] <= Q, name=f"u_ub_{i}")

for i in N:
    for j in V:
        if i != j:
            model.addConstr(
                u[j] >= u[i] + demand[j] - Q * (1 - x[i, j]),
                name=f"load_{i}_{j}"
            )


# ============================================================
# Solver parameters
# ============================================================
model.Params.TimeLimit = TIME_LIMIT
model.Params.MIPGap = MIP_GAP
model.Params.Threads = 0
model.Params.OutputFlag = 1

# Standard Gurobi tuning, still pure baseline
model.Params.MIPFocus = 1
model.Params.Heuristics = 0.2
model.Params.Cuts = 2
model.Params.Presolve = 2


# ============================================================
# Solve
# ============================================================
model.optimize()


# ============================================================
# IIS if infeasible
# ============================================================
if model.Status == GRB.INFEASIBLE:
    print("\nModel is infeasible. Computing IIS...")
    model.computeIIS()

    iis_path = os.path.join(SCRIPT_DIR, "baseline_model_iis.ilp")
    model.write(iis_path)

    print("\nConstraints in IIS:")
    for constr in model.getConstrs():
        if constr.IISConstr:
            print(f"  {constr.ConstrName}")

    print(f"\nIIS written to: {iis_path}")


# ============================================================
# Route extraction
# ============================================================
def extract_routes(x_vars, node_set, depot_node=0):
    chosen = [(i, j) for (i, j) in x_vars.keys() if x_vars[i, j].X > 0.5]

    succ = {}
    for i, j in chosen:
        succ[i] = j

    starts = [j for (i, j) in chosen if i == depot_node]

    routes = []
    for s in starts:
        route = [depot_node, s]
        cur = s

        while cur != depot_node:
            nxt = succ.get(cur, None)
            if nxt is None:
                break
            route.append(nxt)
            cur = nxt

            if len(route) > len(node_set) + 5:
                print("Warning: route extraction stopped due to safety limit.")
                break

        routes.append(route)

    return routes


# ============================================================
# Output solution
# ============================================================
if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
    routes = extract_routes(x, N, depot)

    print("\n" + "=" * 60)
    print("PURE GUROBI BASELINE RESULT")
    print("=" * 60)
    print(f"Status         : {model.Status}")
    print(f"Objective      : {model.ObjVal:.6f}")
    print(f"Best bound     : {model.ObjBound:.6f}")
    print(f"Gap            : {100.0 * model.MIPGap:.4f}%")
    print(f"Runtime        : {model.Runtime:.2f} sec")
    print(f"Vehicles used  : {len(routes)}")
    print("-" * 60)

    total_distance = 0.0
    for r_id, route in enumerate(routes, 1):
        route_dist = 0.0
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            route_dist += c[i, j]
        total_distance += route_dist

        print(f"Route {r_id:2d}: {' -> '.join(map(str, route))} | Dist = {route_dist:.2f}")

    print("-" * 60)
    print(f"Total route distance (without penalty): {total_distance:.2f}")
    print("=" * 60)

else:
    print("\nNo feasible solution found.")
    print("Status code:", model.Status)