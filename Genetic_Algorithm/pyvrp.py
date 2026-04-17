import argparse
import csv
import math
import sys
import os
from pathlib import Path

# WORKAROUND: Allow script to be named pyvrp.py without circular import
# We temporarily remove the current directory from sys.path so we can import the REAL pyvrp library.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
if '' in sys.path:
    sys.path.remove('')

from pyvrp import Model, read
from pyvrp.solve import solve
from pyvrp.stop import MaxRuntime

# Restore sys.path for other imports
sys.path.insert(0, script_dir)

# ──────────────────────────────────────────────
# Helper: rebuild ProblemData with a capped fleet
# ──────────────────────────────────────────────
def _build_model_with_fleet(data, num_vehicles: int) -> Model:
    """Reconstruct a PyVRP Model from existing ProblemData,
    but limit the fleet to `num_vehicles`."""
    m = Model()
    new_locs = []

    # Depot (index 0)
    depot = data.depots()[0]
    m_depot = m.add_depot(
        x        = depot.x,
        y        = depot.y,
        tw_early = depot.tw_early,
        tw_late  = depot.tw_late,
    )
    new_locs.append(m_depot)

    # Clients (indices 1 … n)
    for client in data.clients():
        m_client = m.add_client(
            x                = client.x,
            y                = client.y,
            delivery         = client.delivery,
            service_duration = client.service_duration,
            tw_early         = client.tw_early,
            tw_late          = client.tw_late,
        )
        new_locs.append(m_client)

    # Edges (distances and durations)
    dist_mat = data.distance_matrix(0)
    dur_mat  = data.duration_matrix(0)
    for i in range(data.num_locations):
        for j in range(data.num_locations):
            if i == j: continue
            m.add_edge(new_locs[i], new_locs[j], 
                       distance = int(dist_mat[i, j]),
                       duration = int(dur_mat[i, j]))

    # Vehicle type – same capacity, capped fleet size
    vt = data.vehicle_type(0)
    m.add_vehicle_type(
        num_available = num_vehicles,
        capacity      = vt.capacity,
    )

    return m


# ──────────────────────────────────────────────
# STEP 1 – Minimise the number of vehicles
# ──────────────────────────────────────────────
def step1_minimize_vehicles(data, time_limit: float):
    print("=" * 55)
    print("STEP 1 — Minimise Number of Vehicles")
    print("=" * 55)

    # First, get a feasible solution to find an initial fleet size
    result = solve(data, stop=MaxRuntime(time_limit), display=False)
    if not result.best.is_feasible():
        raise RuntimeError("Could not find any feasible solution to start with.")
    
    best_sol = result.best
    min_k    = best_sol.num_routes()
    print(f"Initial feasible fleet size: {min_k}")

    while min_k > 1:
        candidate_k = min_k - 1
        print(f"\n  Trying {candidate_k} vehicle(s) …", end=" ", flush=True)

        m          = _build_model_with_fleet(data, candidate_k)
        new_data   = m.data()
        new_result = solve(new_data, stop=MaxRuntime(time_limit), display=False)
        new_sol    = new_result.best

        if new_sol.is_feasible():
            print(f"✔  feasible (dist={new_sol.distance() / 10:.2f})")
            best_sol = new_sol
            min_k    = candidate_k
        else:
            print("✘  infeasible — minimum fleet size found.")
            break

    print(f"\n✔ Minimum vehicles found: {min_k}")
    return min_k, best_sol


# ──────────────────────────────────────────────
# STEP 2 – Minimise distance for the fixed min_k
# ──────────────────────────────────────────────
def step2_minimize_distance(data, min_k: int, time_limit: float):
    print("\n" + "=" * 55)
    print(f"STEP 2 — Minimise Distance (Fixed k={min_k})")
    print("=" * 55)

    # Rebuild model with the fixed minimum fleet size
    m    = _build_model_with_fleet(data, min_k)
    data_fixed = m.data()
    
    # Run a thorough optimization
    result = solve(data_fixed, stop=MaxRuntime(time_limit), display=True)
    
    print(f"\n✔ Optimized Distance: {result.best.distance() / 10:.2f}")
    return result.best


# ──────────────────────────────────────────────
# Pretty-print and Save the final solution
# ──────────────────────────────────────────────
def print_solution(sol):
    print("\n" + "=" * 55)
    print("FINAL SOLUTION")
    print("=" * 55)
    print(f"  Vehicles used : {sol.num_routes()}")
    print(f"  Total distance: {sol.distance() / 10:.2f}")  # Unscale
    print()
    for i, route in enumerate(sol.routes(), 1):
        visits = " → ".join(str(v) for v in route)
        print(f"  Route {i:>2}: 0 → {visits} → 0")


def save_output(instance_path: str, sol):
    """Save the instance and solution in a structured folder."""
    import shutil
    
    # Base directory for all instances
    base_dir = Path("dataset")
    base_dir.mkdir(exist_ok=True)

    # 1. Determine the folder name (inst_001, inst_002, ...)
    counter = 1
    while True:
        folder_name = f"inst_{counter:03d}"
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            break
        counter += 1
    
    # 2. Create the folder structure
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Move the main file (instance) to the folder
    # The user requested to MOVE the file instead of copying it.
    dest_name = "data.csv" if instance_path.lower().endswith(".csv") else "main.txt"
    shutil.move(instance_path, folder_path / dest_name)
    
    # 4. Save the solution as solution.txt
    sol_lines = []
    for i, route in enumerate(sol.routes(), 1):
        visits = " ".join(str(v) for v in route)
        sol_lines.append(f"Route {i} : {visits}")
    
    with open(folder_path / "solution.txt", "w") as f:
        f.write("\n".join(sol_lines))
        
    print(f"\n✔ Solution saved in folder: {folder_path}")
    print(f"  - {folder_path}/{dest_name}")
    print(f"  - {folder_path}/solution.txt")


# ──────────────────────────────────────────────
# Instance Reading
# ──────────────────────────────────────────────
def _read_csv(path: Path) -> Model:
    """Read a Solomon CSV instance and return a Model."""
    m = Model()
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    def get_val(row, keys: list, default=0):
        for k in row.keys():
            if k.strip().upper() in [s.upper() for s in keys]:
                return row[k]
        return default

    # First row is depot
    depot = rows[0]
    m.add_depot(
        x        = round(float(get_val(depot, ["XCOORD.", "x"])) * 10),
        y        = round(float(get_val(depot, ["YCOORD.", "y"])) * 10),
        tw_early = round(float(get_val(depot, ["READY TIME", "ready", "tw_start"])) * 10),
        tw_late  = round(float(get_val(depot, ["DUE DATE", "due", "tw_end"])) * 10),
    )

    # Others are clients
    for i, row in enumerate(rows[1:], 1):
        m.add_client(
            x                = round(float(get_val(row, ["XCOORD.", "x"])) * 10),
            y                = round(float(get_val(row, ["YCOORD.", "y"])) * 10),
            delivery         = [int(get_val(row, ["DEMAND", "demand", "q"]))],
            service_duration = round(float(get_val(row, ["SERVICE TIME", "service"])) * 10),
            tw_early         = round(float(get_val(row, ["READY TIME", "ready", "tw_start"])) * 10),
            tw_late          = round(float(get_val(row, ["DUE DATE", "due", "tw_end"])) * 10),
        )
    
    # All-pairs Euclidean distances
    locations = m.locations
    for loc_i in locations:
        for loc_j in locations:
            if loc_i is loc_j: continue
            dist = math.sqrt((loc_i.x - loc_j.x)**2 + (loc_i.y - loc_j.y)**2)
            d_int = round(dist)
            m.add_edge(loc_i, loc_j, distance=d_int, duration=d_int)

    # Standard Solomon capacity
    m.add_vehicle_type(num_available=25, capacity=200)
    return m


def solve_vrptw(filepath: str,
                phase1_time: float = 300.0,
                phase2_time: float = 30.0):
    """
    Solve VRPTW on a Solomon benchmark instance (.txt or .csv).
    """
    print(f"\nLoading instance: {filepath}")
    
    path = Path(filepath)
    if path.suffix.lower() == ".csv":
        model = _read_csv(path)
        data = model.data()
    else:
        # Standard VRPLIB/Solomon TXT
        data = read(filepath)

    print(f"  Customers : {data.num_clients}")
    print(f"  Locations : {data.num_locations}  (incl. depot)")
    print(f"  Vehicles  : {data.vehicle_type(0).num_available} available")
    print(f"  Capacity  : {data.vehicle_type(0).capacity}")

    min_k, initial_sol = step1_minimize_vehicles(data, phase1_time)
    final_sol         = step2_minimize_distance(data, min_k, phase2_time)
    print_solution(final_sol)
    
    # Save the output to a folder
    save_output(filepath, final_sol)

    return final_sol


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch VRPTW solver using PyVRP"
    )
    parser.add_argument("instance",
                        nargs="?",
                        default=None,
                        help="Path to instance file or directory")
    parser.add_argument("--phase1-time", type=float, default=20.0,
                        help="Time limit (s) for Step 1 per k-attempt (default 20)")
    parser.add_argument("--phase2-time", type=float, default=60.0,
                        help="Time limit (s) for Step 2 final optimization (default 60)")
    args = parser.parse_args()

    instance_arg = args.instance
    
    # 1. Collect files to process
    if not instance_arg:
        # Search in CWD and script's directory for all CSV/TXT
        search_dirs = [Path("."), Path(__file__).parent]
        files = []
        for d in search_dirs:
            files.extend(list(d.glob("*.csv")) + list(d.glob("*.txt")))
        # Keep only unique absolute paths and exclude already processed ones
        files = [f.absolute() for f in files if "inst_" not in f.parent.name]
        files = sorted(list(set(files)))
    else:
        p = Path(instance_arg)
        if p.is_dir():
            files = sorted(list(p.glob("*.csv")) + list(p.glob("*.txt")))
        else:
            files = [p]

    if not files:
        print("No instance files (.csv or .txt) found.")
    else:
        print(f"Found {len(files)} instance(s) to process.")

        # 2. Batch processing loop
        for fpath in files:
            try:
                solve_vrptw(str(fpath), args.phase1_time, args.phase2_time)
            except Exception as e:
                print(f"\n❌ Error processing {fpath.name}: {e}")
                continue