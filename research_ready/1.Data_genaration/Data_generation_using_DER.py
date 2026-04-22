
import numpy as np
import pandas as pd
import os
import argparse
import tempfile
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import genextreme as gev_dist
from scipy.stats import weibull_min

from pyvrp import read
from pyvrp.solve import solve
from pyvrp.stop import MaxRuntime

# ============================================================
# Configuration
# ============================================================

SERIES_PARAMS = {
    'C1':  {'Q': 200,  's': 90, 'horizon': 1236, 'n_vehicles_min': 10, 'n_vehicles_max': 25},
    'C2':  {'Q': 200,  's': 90, 'horizon': 3390, 'n_vehicles_min': 10, 'n_vehicles_max': 25},
    'R1':  {'Q': 200,  's': 10, 'horizon': 230,  'n_vehicles_min': 10, 'n_vehicles_max': 25},
    'R2':  {'Q': 200, 's': 10, 'horizon': 1000, 'n_vehicles_min': 10, 'n_vehicles_max': 25},
    'RC1': {'Q': 200,  's': 10, 'horizon': 240,  'n_vehicles_min': 10, 'n_vehicles_max': 25},
    'RC2': {'Q': 200, 's': 10, 'horizon': 960,  'n_vehicles_min': 10, 'n_vehicles_max': 25},
}

DEPOT_X = 40.0
DEPOT_Y = 50.0


# ============================================================
# Demand Sampling
# ============================================================

def sample_demand(series: str) -> int:
    if series.startswith('C'):
        return int(np.random.choice([10, 20, 30, 40]))
    return int(np.random.randint(1, 51))


def scale_demands_to_capacity(demands, vehicle_count, capacity, min_demand=1):
    max_total_capacity = vehicle_count * capacity
    target_utilization = np.random.uniform(0.75, 0.95)
    target_total_demand = max_total_capacity * target_utilization

    current_total = sum(demands)

    if current_total == 0:
        return demands

    scale_factor = target_total_demand / current_total

    scaled = []
    for d in demands:
        val = max(min_demand, int(round(d * scale_factor)))
        scaled.append(min(val, capacity))

    while sum(scaled) > max_total_capacity:
        idx = np.random.randint(len(scaled))
        if scaled[idx] > min_demand:
            scaled[idx] -= 1

    return scaled


# ============================================================
# Coordinate Generation
# ============================================================

def cluster_seeds():
    return np.array([
        [25, 15], [55, 15], [85, 15],
        [25, 50], [55, 50], [85, 50],
        [25, 85], [55, 85], [85, 85],
    ], dtype=float)


def generate_coords(series: str, n: int):
    depot = np.array([[DEPOT_X, DEPOT_Y]])

    if series.startswith('R') and not series.startswith('RC'):
        cust = np.random.uniform(0, 100, (n, 2))

    elif series.startswith('C'):
        seeds = cluster_seeds()
        cust = []
        for _ in range(n):
            seed = seeds[np.random.randint(len(seeds))]
            point = seed + np.random.normal(0, 5, 2)
            cust.append(point)
        cust = np.array(cust)

    else:
        n_cluster = n // 2
        n_random = n - n_cluster

        seeds = cluster_seeds()

        random_points = np.random.uniform(0, 100, (n_random, 2))

        cluster_points = []
        for _ in range(n_cluster):
            seed = seeds[np.random.randint(len(seeds))]
            point = seed + np.random.normal(0, 5, 2)
            cluster_points.append(point)

        cluster_points = np.array(cluster_points)
        cust = np.vstack([random_points, cluster_points])

    coords = np.vstack([depot, cust])
    coords = np.clip(coords, 0, 100)
    coords = np.round(coords, 2)

    seen = set()
    final_customers = []

    for row in coords[1:]:
        key = tuple(row)

        while key in seen or key == tuple(depot[0]):
            row = np.clip(row + np.random.normal(0, 0.5, 2), 0, 100)
            row = np.round(row, 2)
            key = tuple(row)

        seen.add(key)
        final_customers.append(row)

    coords = np.vstack([depot, np.array(final_customers)])
    return coords


# ============================================================
# Half Width Sampling
# ============================================================

def sample_half_width(series: str):
    p = np.random.rand()

    if series == 'C1':
        if p < 3/8:
            return float(beta_dist.rvs(4.06, 5.95, loc=16.05, scale=35.34))
        elif p < 4/8:
            return float(beta_dist.rvs(3.66, 5.33, loc=33.51, scale=67.03))
        elif p < 5/8:
            return float(gamma_dist.rvs(1.52, loc=43.03, scale=12.49))
        elif p < 6/8:
            return float(beta_dist.rvs(3.73, 5.23, loc=66.20, scale=133.38))
        elif p < 7/8:
            return 90.0
        return 180.0

    elif series == 'C2':
        if p < 4/8:
            return 80.0
        elif p < 5/8:
            return 160.0
        elif p < 6/8:
            return 320.0
        elif p < 7/8:
            return float(beta_dist.rvs(3.67, 5.20, loc=133.29, scale=266.06))
        return float(beta_dist.rvs(0.86, 1.41, loc=88.50, scale=547.94))

    elif series == 'R1':
        return float(np.random.choice([5, 15, 30, 45, 60]))

    elif series == 'R2':
        return float(np.random.choice([60, 120, 180, 240]))

    elif series == 'RC1':
        return float(np.random.choice([15, 30, 45, 60]))

    elif series == 'RC2':
        return float(np.random.choice([60, 120, 180, 240]))

    return 30.0


# ============================================================
# Center Estimation
# ============================================================

def estimate_center_c_series(dist_from_depot, horizon, service_time):
    earliest = dist_from_depot
    latest = horizon - dist_from_depot - service_time

    if earliest >= latest:
        return earliest

    return float(np.random.uniform(
        earliest,
        min(latest, earliest + (latest - earliest) * 0.6)
    ))


# ============================================================
# Instance Generation
# ============================================================

def generate_instance(series='R1', n_customers=100, density=1.0, min_tw_width=5):
    params = SERIES_PARAMS[series]

    capacity = params['Q']
    service_time = params['s']
    horizon = params['horizon']

    n_vehicles = np.random.randint(
        params['n_vehicles_min'],
        params['n_vehicles_max'] + 1
    )

    coords = generate_coords(series, n_customers)
    depot = coords[0]

    raw_demands = [sample_demand(series) for _ in range(n_customers)]

    min_demand = 10 if series.startswith('C') else 1

    scaled_demands = scale_demands_to_capacity(
        raw_demands,
        n_vehicles,
        capacity,
        min_demand=min_demand
    )

    rows = []

    rows.append({
        'CUST_NO': 0,
        'XCOORD.': depot[0],
        'YCOORD.': depot[1],
        'DEMAND': 0,
        'READY TIME': 0,
        'DUE DATE': horizon,
        'SERVICE TIME': 0,
    })

    restricted = np.random.rand(n_customers) < density

    for i in range(1, n_customers + 1):
        x, y = coords[i]
        dist = np.sqrt((x - depot[0]) ** 2 + (y - depot[1]) ** 2)

        earliest_arrival = dist
        latest_departure = horizon - dist - service_time

        if earliest_arrival >= latest_departure:
            ready = 0
            due = horizon
        elif restricted[i - 1]:
            if series.startswith('C'):
                center = estimate_center_c_series(dist, horizon, service_time)
            else:
                center = np.random.uniform(earliest_arrival, latest_departure)

            hw = abs(sample_half_width(series))

            ready = max(earliest_arrival, center - hw)
            due = min(latest_departure + service_time, center + hw)

            if due - ready < min_tw_width:
                due = min(horizon, ready + min_tw_width)
                ready = max(0, due - min_tw_width)

        else:
            ready = 0
            due = horizon

        rows.append({
            'CUST_NO': i,
            'XCOORD.': round(float(x), 2),
            'YCOORD.': round(float(y), 2),
            'DEMAND': int(scaled_demands[i - 1]),
            'READY TIME': round(float(ready), 2),
            'DUE DATE': round(float(due), 2),
            'SERVICE TIME': service_time,
        })

    df = pd.DataFrame(rows)
    df.attrs['n_vehicles'] = int(n_vehicles)
    df.attrs['capacity'] = int(capacity)

    return df


# ============================================================
# Basic Feasibility Check
# ============================================================

def is_feasible(df, series):
    horizon = SERIES_PARAMS[series]['horizon']

    for _, row in df.iloc[1:].iterrows():
        ready = row['READY TIME']
        due = row['DUE DATE']
        service = row['SERVICE TIME']
        x = row['XCOORD.']
        y = row['YCOORD.']

        if due < ready:
            return False

        dist = np.sqrt((x - DEPOT_X) ** 2 + (y - DEPOT_Y) ** 2)

        if ready + service + dist > horizon:
            return False

        if due > horizon:
            return False

    return True


# ============================================================
# Solver Validation
# ============================================================

def validate_with_solver(df, series, time_limit=30):
    """
    Validation using the actual PyVRP solver.
    Builds the model in memory and checks if it's solvable within the time limit.
    """
    from pyvrp import Model
    import math

    try:
        m = Model()
        depot_row = df.iloc[0]
        
        # Add Depot
        # PyVRP uses integers for coordinates/time, so we scale by 10 for precision
        m.add_depot(
            x=round(float(depot_row['XCOORD.']) * 10),
            y=round(float(depot_row['YCOORD.']) * 10),
            tw_early=round(float(depot_row['READY TIME']) * 10),
            tw_late=round(float(depot_row['DUE DATE']) * 10),
        )

        # Add Clients
        for _, row in df.iloc[1:].iterrows():
            m.add_client(
                x=round(float(row['XCOORD.']) * 10),
                y=round(float(row['YCOORD.']) * 10),
                delivery=[int(row['DEMAND'])],
                service_duration=round(float(row['SERVICE TIME']) * 10),
                tw_early=round(float(row['READY TIME']) * 10),
                tw_late=round(float(row['DUE DATE']) * 10),
            )

        # Add Edges (Euclidean)
        locations = m.locations
        for loc_i in locations:
            for loc_j in locations:
                if loc_i is loc_j: continue
                dist = math.sqrt((loc_i.x - loc_j.x)**2 + (loc_i.y - loc_j.y)**2)
                d_int = round(dist)
                m.add_edge(loc_i, loc_j, distance=d_int, duration=d_int)

        # Add Vehicles (Standardized to 25/200)
        m.add_vehicle_type(num_available=25, capacity=200)

        # Solve
        res = solve(m.data(), stop=MaxRuntime(time_limit), display=False)
        return res.best.is_feasible()

    except Exception as e:
        print(f"  (Validation Error: {e})")
        return False


# ============================================================
# Batch Generation
# ============================================================

def generate_batch(
    output_dir='dataset',
    selected_series=None,
    density_counts=None,
    n_customers=100,
    seed=42,
    solver_time_limit=30,
):
    """
    density_counts: dict mapping density (float) -> number of instances to generate.
    Example: {0.25: 15, 0.50: 20, 0.75: 30, 1.00: 35}
    """
    np.random.seed(seed)

    if selected_series is None:
        selected_series = list(SERIES_PARAMS.keys())

    if density_counts is None:
        density_counts = {0.25: 10, 0.50: 10, 0.75: 10, 1.00: 10}

    densities = sorted(density_counts.keys())
    os.makedirs(output_dir, exist_ok=True)

    total_saved = 0
    total_target = sum(
        len(selected_series) * density_counts[d] for d in densities
    )

    # Print the distribution plan
    print(f"  Density distribution plan:")
    for d in densities:
        print(f"    d{int(d*100):3d} → {density_counts[d]:4d} instances per series")
    print(f"  Grand total target: {total_target} instances")
    print()

    for series in selected_series:
        # Mirror data/ folder structure: save into a series subfolder
        series_dir = os.path.join(output_dir, series)
        os.makedirs(series_dir, exist_ok=True)

        for density in densities:
            n_instances_for_density = density_counts[density]
            density_tag = int(density * 100)
            saved_for_group = 0
            attempts = 0

            while saved_for_group < n_instances_for_density:
                attempts += 1

                progress_pct = (total_saved / total_target * 100) if total_target > 0 else 0
                progress_info = f"[{total_saved:04d}/{total_target:04d}] {progress_pct:5.1f}%"

                print(f"  {progress_info} | Attempt {attempts}: Generating {series}/d{density_tag} ...", end="\r")

                if attempts > n_instances_for_density * 100:
                    print(f"\n❌ Too many failed attempts for {series} density={density}")
                    break

                df = generate_instance(
                    series=series,
                    n_customers=n_customers,
                    density=density,
                )

                basic_ok = is_feasible(df, series)
                if not basic_ok:
                    continue

                print(f"  {progress_info} | Attempt {attempts}: Validating with solver ({solver_time_limit}s) ...", end="\r")
                solver_ok = validate_with_solver(
                    df,
                    series=series,
                    time_limit=solver_time_limit,
                )
                if not solver_ok:
                    continue

                saved_for_group += 1
                total_saved += 1
                print(" " * 100, end="\r")  # Clear the line

                filename = f"{series}_n{n_customers}_d{density_tag}_{saved_for_group:04d}.csv"
                filepath = os.path.join(series_dir, filename)

                df.to_csv(filepath, index=False)

                new_pct = (total_saved / total_target * 100) if total_target > 0 else 0
                print(
                    f"  [{total_saved:04d}/{total_target:04d}] {new_pct:5.1f}% | "
                    f"Saved {series}/{filename} | Vehicles={df.attrs['n_vehicles']}"
                )

    print(f"\nFinished. Total saved instances: {total_saved}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dynamic DER-Solomon Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 100 instances/series, distributed 15/20/30/35 across d25/d50/d75/d100
  python Data_generation_using_DER.py --n_instances 100

  # Custom total with custom weights
  python Data_generation_using_DER.py --n_instances 200 --density_weights 10 20 30 40

  # Only R-series, 50 instances/series
  python Data_generation_using_DER.py --series R1 R2 --n_instances 50
"""
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        help='Root folder to save generated instances (series subfolders created automatically)'
    )

    parser.add_argument(
        '--series',
        nargs='+',
        default=['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'],
        choices=['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'],
        help='Which series to generate'
    )

    parser.add_argument(
        '--n_customers',
        type=int,
        default=100,
        help='Number of customers per instance'
    )

    parser.add_argument(
        '--n_instances',
        type=int,
        default=10,
        help='Total instances per series (distributed across densities via --density_weights)'
    )

    parser.add_argument(
        '--density_weights',
        nargs=4,
        type=float,
        default=[15.0, 20.0, 30.0, 35.0],
        metavar=('W_D25', 'W_D50', 'W_D75', 'W_D100'),
        help='Relative weights for d25/d50/d75/d100 (default: 15 20 30 35)'
    )

    parser.add_argument(
        '--densities',
        nargs='+',
        type=float,
        default=[0.25, 0.50, 0.75, 1.00],
        help='Density values to generate (must match number of density_weights)'
    )

    parser.add_argument(
        '--solver_time',
        type=int,
        default=5,
        help='Solver validation time in seconds'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Build per-density counts from total n_instances + weights
    densities = args.densities
    weights   = args.density_weights
    if len(weights) != len(densities):
        raise ValueError(
            f"--density_weights has {len(weights)} values but --densities has {len(densities)}. "
            "They must match."
        )

    total_weight = sum(weights)
    raw_counts   = [args.n_instances * w / total_weight for w in weights]

    # Round while preserving exact total (largest-remainder method)
    floors  = [int(c) for c in raw_counts]
    remainders = sorted(
        range(len(raw_counts)), key=lambda i: -(raw_counts[i] - floors[i])
    )
    leftover = args.n_instances - sum(floors)
    for i in remainders[:leftover]:
        floors[i] += 1

    density_counts = dict(zip(densities, floors))

    print('=' * 60)
    print('DER-Solomon Dynamic Instance Generator')
    print('=' * 60)
    print(f'Series           : {args.series}')
    print(f'Customers        : {args.n_customers}')
    print(f'Total/Series     : {args.n_instances}')
    print(f'Density weights  : d25={weights[0]}% | d50={weights[1]}% | d75={weights[2]}% | d100={weights[3]}%')
    print(f'Per-density counts (per series):')
    for d, n in density_counts.items():
        print(f'    d{int(d*100):3d} → {n:4d} instances  ({n/args.n_instances*100:.1f}%)')
    print(f'Solver Time      : {args.solver_time}s')
    print(f'Output Folder    : {args.output_dir}')
    print('=' * 60)

    generate_batch(
        output_dir=args.output_dir,
        selected_series=args.series,
        density_counts=density_counts,
        n_customers=args.n_customers,
        seed=args.seed,
        solver_time_limit=args.solver_time,
    )
