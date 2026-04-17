# Full Improved VRPTW Generator Code

import numpy as np
import pandas as pd
import os
import argparse
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import genextreme as gev_dist
from scipy.stats import weibull_min

SERIES_PARAMS = {
    'C1':  {'Q': 200,  's': 90, 'horizon': 1236, 'n_vehicles_min': 8,  'n_vehicles_max': 13},
    'C2':  {'Q': 700,  's': 90, 'horizon': 3390, 'n_vehicles_min': 2,  'n_vehicles_max': 6},
    'R1':  {'Q': 200,  's': 10, 'horizon': 230,  'n_vehicles_min': 10, 'n_vehicles_max': 19},
    'R2':  {'Q': 1000, 's': 10, 'horizon': 1000, 'n_vehicles_min': 2,  'n_vehicles_max': 6},
    'RC1': {'Q': 200,  's': 10, 'horizon': 240,  'n_vehicles_min': 10, 'n_vehicles_max': 16},
    'RC2': {'Q': 1000, 's': 10, 'horizon': 960,  'n_vehicles_min': 2,  'n_vehicles_max': 6},
}

DEPOT_X, DEPOT_Y = 40.0, 50.0


def sample_demand(series: str) -> int:
    if series.startswith('C'):
        return int(np.random.choice([10, 20, 30, 40]))
    return int(np.random.randint(1, 51))


def _cluster_seeds() -> np.ndarray:
    return np.array([
        [25, 15], [55, 15], [85, 15],
        [25, 50], [55, 50], [85, 50],
        [25, 85], [55, 85], [85, 85],
    ], dtype=float)


def generate_coords(series: str, n: int) -> np.ndarray:
    depot = np.array([[DEPOT_X, DEPOT_Y]])

    if series.startswith('R') and not series.startswith('RC'):
        cust = np.random.uniform(0, 100, (n, 2))

    elif series.startswith('C'):
        seeds = _cluster_seeds()
        cust = []
        for _ in range(n):
            seed = seeds[np.random.randint(len(seeds))]
            cust.append(seed + np.random.normal(0, 5, 2))
        cust = np.array(cust)

    else:
        n_clust = n // 2
        n_rand = n - n_clust

        seeds = _cluster_seeds()
        rand_part = np.random.uniform(0, 100, (n_rand, 2))

        clust_part = []
        for _ in range(n_clust):
            seed = seeds[np.random.randint(len(seeds))]
            clust_part.append(seed + np.random.normal(0, 5, 2))

        cust = np.vstack([rand_part, np.array(clust_part)])

    coords = np.vstack([depot, cust])
    return np.clip(coords, 0, 100)


def sample_half_width(series: str) -> float:
    if series in ['C1', 'RC1', 'R1']:
        return float(np.random.choice([5, 10, 15, 20, 30, 60]))
    return float(np.random.choice([30, 60, 120, 180, 240]))


def estimate_center_c_series(dist_from_depot, horizon, service_time):
    earliest = dist_from_depot
    latest = horizon - dist_from_depot - service_time

    if earliest >= latest:
        return earliest

    return float(np.random.uniform(
        earliest,
        min(latest, earliest + (latest - earliest) * 0.6)
    ))


def generate_instance(
    series: str,
    n_customers: int = 100,
    density: float = 1.0,
    min_tw_width: float = 1.0,
    n_vehicles: int | None = None,
):
    params = SERIES_PARAMS[series]
    horizon = float(params['horizon'])
    svc = float(params['s'])
    capacity = params['Q']

    if n_vehicles is None:
        n_vehicles = int(np.random.randint(
            params['n_vehicles_min'],
            params['n_vehicles_max'] + 1
        ))

    coords = generate_coords(series, n_customers)
    depot = coords[0]

    rows = []
    customer_demands = []

    rows.append({
        'CUST_NO': 0,
        'XCOORD.': round(depot[0], 2),
        'YCOORD.': round(depot[1], 2),
        'DEMAND': 0,
        'READY TIME': 0.0,
        'DUE DATE': horizon,
        'SERVICE TIME': 0.0,
    })

    is_restricted = np.random.rand(n_customers) < density

    for i in range(1, n_customers + 1):
        cx, cy = coords[i]
        dist = float(np.sqrt((cx - depot[0]) ** 2 + (cy - depot[1]) ** 2))

        earliest_arrival = dist
        latest_departure = horizon - dist - svc

        if earliest_arrival >= latest_departure:
            ready = 0.0
            due = horizon

        elif is_restricted[i - 1]:
            if series.startswith('C'):
                center = estimate_center_c_series(dist, horizon, svc)
            else:
                center = float(np.random.uniform(earliest_arrival, latest_departure))

            hw = sample_half_width(series)

            ready = max(earliest_arrival, center - hw)
            due = min(latest_departure + svc, center + hw)

            if due - ready < min_tw_width:
                due = ready + min_tw_width

            ready = max(0.0, ready)
            due = min(horizon, due)

        else:
            ready = 0.0
            due = horizon

        demand = sample_demand(series)
        customer_demands.append(demand)

        rows.append({
            'CUST_NO': i,
            'XCOORD.': round(cx, 2),
            'YCOORD.': round(cy, 2),
            'DEMAND': demand,
            'READY TIME': round(ready, 2),
            'DUE DATE': round(due, 2),
            'SERVICE TIME': svc,
        })

    df = pd.DataFrame(rows)

    total_demand = sum(customer_demands)
    total_capacity = n_vehicles * capacity

    if total_demand > total_capacity * 0.85:
        scale_factor = (total_capacity * 0.85) / total_demand

        scaled_demands = []
        for d in customer_demands:
            scaled_demands.append(max(1, int(round(d * scale_factor))))

        df.loc[1:, 'DEMAND'] = scaled_demands

    df.attrs['n_vehicles'] = n_vehicles
    df.attrs['capacity'] = capacity
    df.attrs['total_capacity'] = total_capacity
    df.attrs['total_demand'] = int(df.loc[1:, 'DEMAND'].sum())

    return df


def is_feasible(df: pd.DataFrame, series: str) -> bool:
    horizon = float(SERIES_PARAMS[series]['horizon'])
    capacity = df.attrs['capacity']
    n_vehicles = df.attrs['n_vehicles']

    cust = df.iloc[1:]

    if (cust['DUE DATE'] < cust['READY TIME']).any():
        return False

    if (cust['READY TIME'] < 0).any():
        return False

    if (cust['DUE DATE'] > horizon).any():
        return False

    if (cust['DEMAND'] > capacity).any():
        return False

    total_demand = cust['DEMAND'].sum()
    total_capacity = n_vehicles * capacity

    if total_demand > total_capacity:
        return False

    return True


def generate_batch(
    output_dir: str,
    series_list=None,
    n_per_series: int = 1,
    n_customers: int = 100,
    densities=None,
    seed: int = 42,
):
    np.random.seed(seed)

    if series_list is None:
        series_list = list(SERIES_PARAMS.keys())

    if densities is None:
        densities = [0.25, 0.50, 0.75, 1.00]

    os.makedirs(output_dir, exist_ok=True)

    total = 0

    for series in series_list:
        for density in densities:
            d_tag = int(density * 100)

            for idx in range(1, n_per_series + 1):
                for _ in range(10):
                    df = generate_instance(
                        series=series,
                        n_customers=n_customers,
                        density=density,
                    )

                    if is_feasible(df, series):
                        break

                filename = f"{series}_n{n_customers}_d{d_tag}_{idx:04d}.csv"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'w') as fout:
                    fout.write(
                        f"# n_vehicles={df.attrs['n_vehicles']} "
                        f"capacity={df.attrs['capacity']} "
                        f"total_capacity={df.attrs['total_capacity']} "
                        f"total_demand={df.attrs['total_demand']}\n"
                    )
                    df.to_csv(fout, index=False)

                total += 1
                print(f"Saved {filename}")

    print(f"\nGenerated {total} instances in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--series', nargs='+', default=None)
    parser.add_argument('--n_per_series', type=int, default=1)
    parser.add_argument('--n_customers', type=int, default=100)
    parser.add_argument('--densities', nargs='+', type=float, default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    generate_batch(
        output_dir=args.output_dir,
        series_list=args.series,
        n_per_series=args.n_per_series,
        n_customers=args.n_customers,
        densities=args.densities,
        seed=args.seed,
    )
