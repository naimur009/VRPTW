#!/usr/bin/env python3
"""
Safer Gurobi solver for VRPTW on pruned graphs.

Main safety upgrades:
1. Time-feasible safe-core edge protection
2. Always preserve depot connectivity
3. Keep high-probability arcs
4. Keep top-k feasible arcs by probability and by distance
5. Adaptive protection for tight time-window customers
6. Feasibility audit before solve
7. Safer repair: add feasible fallback arcs first
8. log_to_console now correctly controls OutputFlag

Usage:
    python3 solve_pruned_instances_safe.py
"""

import os
import sys
import time
import math
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# Helpers
# ============================================================

def euclidean(p1, p2) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def get_col(df: pd.DataFrame, candidates: List[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def build_full_distance_dict(nodes_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    coords = {
        int(r["node_id"]): (float(r["x"]), float(r["y"]))
        for _, r in nodes_df.iterrows()
    }
    V = list(coords.keys())
    dist = {}
    for i in V:
        for j in V:
            if i != j:
                dist[(i, j)] = euclidean(coords[i], coords[j])
    return dist


def compute_route_distance(route: List[int], dist: Dict[Tuple[int, int], float]) -> float:
    total = 0.0
    for k in range(len(route) - 1):
        total += dist.get((route[k], route[k + 1]), 0.0)
    return total


def extract_routes(used_arcs: List[Tuple[int, int]], depot: int) -> List[List[int]]:
    succ = {}
    depot_starts = []

    for i, j in used_arcs:
        if i == depot:
            depot_starts.append(j)
        succ[i] = j

    routes = []
    for start in depot_starts:
        route = [depot]
        cur = start
        seen = {depot}

        while True:
            route.append(cur)

            if cur == depot:
                break
            if cur in seen:
                break
            seen.add(cur)

            if cur not in succ:
                break

            cur = succ[cur]

        routes.append(route)

    return routes


# ============================================================
# VRPTW feasibility helpers
# ============================================================

def is_time_feasible(
    i: int,
    j: int,
    a: Dict[int, float],
    b: Dict[int, float],
    s: Dict[int, float],
    dist: Dict[Tuple[int, int], float]
) -> bool:
    """
    Basic arc time feasibility:
        earliest possible departure from i + travel <= due time at j
    """
    if i == j:
        return False
    return a[i] + s[i] + dist[(i, j)] <= b[j]


def temporal_slack(
    i: int,
    j: int,
    a: Dict[int, float],
    b: Dict[int, float],
    s: Dict[int, float],
    dist: Dict[Tuple[int, int], float]
) -> float:
    return b[j] - (a[i] + s[i] + dist[(i, j)])


def compute_window_widths(
    V: List[int],
    a: Dict[int, float],
    b: Dict[int, float]
) -> Dict[int, float]:
    return {i: float(b[i] - a[i]) for i in V}


def classify_tight_customers(
    C: List[int],
    a: Dict[int, float],
    b: Dict[int, float],
    tight_quantile: float = 0.25
) -> Set[int]:
    widths = np.array([b[i] - a[i] for i in C], dtype=float)
    if len(widths) == 0:
        return set()
    threshold = float(np.quantile(widths, tight_quantile))
    return {i for i in C if (b[i] - a[i]) <= threshold}


# ============================================================
# Safe-core construction
# ============================================================

def build_candidate_arc_rows(edges_df: pd.DataFrame) -> Dict[Tuple[int, int], Dict]:
    rows = {}
    for _, r in edges_df.iterrows():
        i, j = int(r["from"]), int(r["to"])
        if i != j:
            rows[(i, j)] = dict(r)
    return rows


def build_default_arc_row(
    i: int,
    j: int,
    full_dist: Dict[Tuple[int, int], float],
    prob_val: float = 0.0,
    reason: str = "repair_added"
) -> Dict:
    return {
        "from": i,
        "to": j,
        "distance": full_dist[(i, j)],
        "prob": prob_val,
        "keep_for_solver": 1,
        "added_by_repair": 1,
        "kept_reason": reason,
    }


def protected_safe_core_edges(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    depot: int,
    a: Dict[int, float],
    b: Dict[int, float],
    s: Dict[int, float],
    full_dist: Dict[Tuple[int, int], float],
    strong_keep_prob: float = 0.70,
    soft_keep_prob: float = 0.35,
    top_k_prob: int = 3,
    top_k_dist: int = 3,
    tight_extra_k: int = 2,
) -> Set[Tuple[int, int]]:
    """
    Build a 'safe core' of protected arcs.

    Keep an arc if any of:
    - depot related
    - high probability
    - in top-k feasible outgoing/incoming by probability
    - in top-k feasible outgoing/incoming by distance
    - extra protection for tight-window customers
    """
    V = [int(v) for v in nodes_df["node_id"].tolist()]
    C = [v for v in V if v != depot]
    tight_customers = classify_tight_customers(C, a, b)

    # Use candidate arcs from scored/pruned graph if present
    arc_rows = build_candidate_arc_rows(edges_df)
    candidate_arcs = set(arc_rows.keys())

    protected = set()

    # Always keep depot connectivity in both directions
    for c in C:
        protected.add((depot, c))
        protected.add((c, depot))

    # Evaluate feasible candidate arcs
    feasible_arcs = []
    for i, j in candidate_arcs:
        if is_time_feasible(i, j, a, b, s, full_dist):
            feasible_arcs.append((i, j))

    # High-probability arcs
    for i, j in feasible_arcs:
        p = float(arc_rows[(i, j)].get("prob", 0.0))
        if p >= strong_keep_prob:
            protected.add((i, j))

    # Per node protection
    for i in C:
        extra = tight_extra_k if i in tight_customers else 0
        k_prob_i = top_k_prob + extra
        k_dist_i = top_k_dist + extra

        # Feasible outgoing from i
        feasible_out = [(ii, j) for (ii, j) in feasible_arcs if ii == i]
        feasible_in = [(j, ii) for (j, ii) in feasible_arcs if ii == i]

        # Top-k by probability
        feasible_out_by_prob = sorted(
            feasible_out,
            key=lambda arc: (
                float(arc_rows[arc].get("prob", 0.0)),
                -full_dist[(arc[0], arc[1])]
            ),
            reverse=True
        )
        feasible_in_by_prob = sorted(
            feasible_in,
            key=lambda arc: (
                float(arc_rows[arc].get("prob", 0.0)),
                -full_dist[(arc[0], arc[1])]
            ),
            reverse=True
        )

        # Top-k by distance
        feasible_out_by_dist = sorted(
            feasible_out,
            key=lambda arc: (
                full_dist[(arc[0], arc[1])],
                -float(arc_rows[arc].get("prob", 0.0))
            )
        )
        feasible_in_by_dist = sorted(
            feasible_in,
            key=lambda arc: (
                full_dist[(arc[0], arc[1])],
                -float(arc_rows[arc].get("prob", 0.0))
            )
        )

        for arc in feasible_out_by_prob[:k_prob_i]:
            protected.add(arc)
        for arc in feasible_in_by_prob[:k_prob_i]:
            protected.add(arc)
        for arc in feasible_out_by_dist[:k_dist_i]:
            protected.add(arc)
        for arc in feasible_in_by_dist[:k_dist_i]:
            protected.add(arc)

        # Soft threshold: keep moderate-prob feasible arcs if tight-window node
        if i in tight_customers:
            for arc in feasible_out:
                if float(arc_rows[arc].get("prob", 0.0)) >= soft_keep_prob:
                    protected.add(arc)
            for arc in feasible_in:
                if float(arc_rows[arc].get("prob", 0.0)) >= soft_keep_prob:
                    protected.add(arc)

    return protected


# ============================================================
# Feasibility audit
# ============================================================

def audit_graph_feasibility(
    V: List[int],
    C: List[int],
    depot: int,
    arcs: Set[Tuple[int, int]],
    a: Dict[int, float],
    b: Dict[int, float],
    s: Dict[int, float],
    dist: Dict[Tuple[int, int], float]
) -> Dict:
    report = {
        "ok": True,
        "missing_out": [],
        "missing_in": [],
        "missing_from_depot": [],
        "missing_to_depot": [],
    }

    def feasible_out(i):
        return [j for (ii, j) in arcs if ii == i and is_time_feasible(ii, j, a, b, s, dist)]

    def feasible_in(i):
        return [j for (j, ii) in arcs if ii == i and is_time_feasible(j, ii, a, b, s, dist)]

    for i in C:
        if len(feasible_out(i)) == 0:
            report["ok"] = False
            report["missing_out"].append(i)
        if len(feasible_in(i)) == 0:
            report["ok"] = False
            report["missing_in"].append(i)
        if (depot, i) not in arcs or not is_time_feasible(depot, i, a, b, s, dist):
            report["ok"] = False
            report["missing_from_depot"].append(i)
        if (i, depot) not in arcs or not is_time_feasible(i, depot, a, b, s, dist):
            report["ok"] = False
            report["missing_to_depot"].append(i)

    return report


# ============================================================
# Safe graph repair
# ============================================================

def repair_pruned_graph_safer(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    depot: int,
    a: Dict[int, float],
    b: Dict[int, float],
    s: Dict[int, float],
    strong_keep_prob: float = 0.70,
    soft_keep_prob: float = 0.35,
    top_k_prob: int = 3,
    top_k_dist: int = 3,
    min_degree: int = 3,
    knn_repair_k: int = 4,
    tight_extra_k: int = 2,
) -> pd.DataFrame:
    """
    Safer repair strategy:
    1. Start from existing keep_for_solver edges if present
    2. Build protected safe-core edges
    3. Keep only time-feasible protected/candidate edges
    4. Ensure depot links
    5. Ensure min feasible in/out degree
    6. Repair using feasible high-score and feasible nearest neighbors
    """
    edges_df = edges_df.copy()
    if "keep_for_solver" in edges_df.columns:
        base_df = edges_df[edges_df["keep_for_solver"] == 1].copy()
    else:
        base_df = edges_df.copy()

    base_df = base_df[base_df["from"] != base_df["to"]].copy()

    V = [int(v) for v in nodes_df["node_id"].tolist()]
    C = [v for v in V if v != depot]
    full_dist = build_full_distance_dict(nodes_df)
    tight_customers = classify_tight_customers(C, a, b)

    # Candidate rows from original scored graph, not only filtered graph
    original_arc_rows = build_candidate_arc_rows(edges_df)
    base_arc_rows = build_candidate_arc_rows(base_df)

    # Protected safe-core from original scored graph
    protected = protected_safe_core_edges(
        nodes_df=nodes_df,
        edges_df=edges_df,
        depot=depot,
        a=a,
        b=b,
        s=s,
        full_dist=full_dist,
        strong_keep_prob=strong_keep_prob,
        soft_keep_prob=soft_keep_prob,
        top_k_prob=top_k_prob,
        top_k_dist=top_k_dist,
        tight_extra_k=tight_extra_k,
    )

    # Start with time-feasible base arcs
    existing_rows = {}
    for (i, j), row in base_arc_rows.items():
        if is_time_feasible(i, j, a, b, s, full_dist):
            row = dict(row)
            row["distance"] = float(row.get("distance", full_dist[(i, j)]))
            row["keep_for_solver"] = 1
            row["added_by_repair"] = int(row.get("added_by_repair", 0))
            row["kept_reason"] = row.get("kept_reason", "base_time_feasible")
            existing_rows[(i, j)] = row

    existing = set(existing_rows.keys())

    def add_arc(i: int, j: int, reason: str = "repair_added", prob_val: float = None):
        if i == j:
            return
        if (i, j) in existing:
            return
        if not is_time_feasible(i, j, a, b, s, full_dist):
            return

        if prob_val is None:
            if (i, j) in original_arc_rows:
                prob_val = float(original_arc_rows[(i, j)].get("prob", 0.0))
            else:
                prob_val = 0.0

        existing_rows[(i, j)] = build_default_arc_row(
            i=i,
            j=j,
            full_dist=full_dist,
            prob_val=prob_val,
            reason=reason,
        )
        existing.add((i, j))

    # Add protected safe-core arcs
    for i, j in protected:
        add_arc(i, j, reason="safe_core")

    # Always ensure depot arcs if time-feasible
    for c in C:
        add_arc(depot, c, reason="depot_connectivity")
        add_arc(c, depot, reason="depot_connectivity")

    def out_neighbors(i: int) -> List[int]:
        return [j for (ii, j) in existing if ii == i]

    def in_neighbors(i: int) -> List[int]:
        return [j for (j, ii) in existing if ii == i]

    def feasible_out_candidates(i: int) -> List[int]:
        return [j for j in V if j != i and is_time_feasible(i, j, a, b, s, full_dist)]

    def feasible_in_candidates(i: int) -> List[int]:
        return [j for j in V if j != i and is_time_feasible(j, i, a, b, s, full_dist)]

    def get_prob(i: int, j: int) -> float:
        if (i, j) in original_arc_rows:
            return float(original_arc_rows[(i, j)].get("prob", 0.0))
        return 0.0

    # Adaptive degree protection
    for i in C:
        extra = tight_extra_k if i in tight_customers else 0
        target_out = max(min_degree + extra, knn_repair_k)
        target_in = max(min_degree + extra, knn_repair_k)

        current_out = set(out_neighbors(i))
        current_in = set(in_neighbors(i))

        # Feasible outgoing candidates not yet kept
        out_cands = [j for j in feasible_out_candidates(i) if j not in current_out]
        in_cands = [j for j in feasible_in_candidates(i) if j not in current_in]

        # First prefer high-probability feasible arcs
        out_by_prob = sorted(
            out_cands,
            key=lambda j: (get_prob(i, j), -full_dist[(i, j)]),
            reverse=True
        )
        in_by_prob = sorted(
            in_cands,
            key=lambda j: (get_prob(j, i), -full_dist[(j, i)]),
            reverse=True
        )

        # Then prefer near feasible arcs
        out_by_dist = sorted(
            out_cands,
            key=lambda j: (full_dist[(i, j)], -get_prob(i, j))
        )
        in_by_dist = sorted(
            in_cands,
            key=lambda j: (full_dist[(j, i)], -get_prob(j, i))
        )

        need_out = max(0, target_out - len(current_out))
        need_in = max(0, target_in - len(current_in))

        used_j = set()
        for j in out_by_prob:
            if need_out <= 0:
                break
            if j in used_j:
                continue
            add_arc(i, j, reason="repair_prob_out")
            used_j.add(j)
            need_out -= 1

        for j in out_by_dist:
            if need_out <= 0:
                break
            if j in used_j:
                continue
            add_arc(i, j, reason="repair_dist_out")
            used_j.add(j)
            need_out -= 1

        used_j = set()
        for j in in_by_prob:
            if need_in <= 0:
                break
            if j in used_j:
                continue
            add_arc(j, i, reason="repair_prob_in")
            used_j.add(j)
            need_in -= 1

        for j in in_by_dist:
            if need_in <= 0:
                break
            if j in used_j:
                continue
            add_arc(j, i, reason="repair_dist_in")
            used_j.add(j)
            need_in -= 1

    repaired_df = pd.DataFrame(list(existing_rows.values()))
    repaired_df = repaired_df.drop_duplicates(subset=["from", "to"]).reset_index(drop=True)

    if "distance" not in repaired_df.columns:
        repaired_df["distance"] = repaired_df.apply(
            lambda r: full_dist[(int(r["from"]), int(r["to"]))], axis=1
        )
    if "prob" not in repaired_df.columns:
        repaired_df["prob"] = 0.0
    if "keep_for_solver" not in repaired_df.columns:
        repaired_df["keep_for_solver"] = 1
    if "added_by_repair" not in repaired_df.columns:
        repaired_df["added_by_repair"] = 0
    if "kept_reason" not in repaired_df.columns:
        repaired_df["kept_reason"] = "unknown"

    return repaired_df


# ============================================================
# Warm start heuristic
# ============================================================

def build_greedy_warm_start(
    V: List[int],
    C: List[int],
    depot: int,
    arcs: Set[Tuple[int, int]],
    demand: Dict[int, float],
    a: Dict[int, float],
    b: Dict[int, float],
    s: Dict[int, float],
    dist: Dict[Tuple[int, int], float],
    prob: Dict[Tuple[int, int], float],
    vehicle_capacity: float,
    max_vehicles: int
) -> List[Tuple[int, int]]:
    """
    Construct a simple feasible-ish greedy warm start.
    It may not always cover all customers, but often helps more than threshold starts.
    """
    unserved = set(C)
    used_arcs = []

    vehicles_used = 0
    while unserved and vehicles_used < max_vehicles:
        cur = depot
        cur_time = max(0.0, a[depot])
        cur_load = 0.0
        route = []

        while True:
            candidates = []
            for j in list(unserved):
                if (cur, j) not in arcs:
                    continue
                if cur_load + demand[j] > vehicle_capacity:
                    continue

                arrival = cur_time + (s[cur] if cur != depot else 0.0) + dist[(cur, j)]
                start_service = max(arrival, a[j])
                if start_service > b[j]:
                    continue

                back_ok = (j, depot) in arcs and is_time_feasible(j, depot, a, b, s, dist)
                if not back_ok:
                    continue

                slack = temporal_slack(cur, j, a, b, s, dist)
                score = (
                    10.0 * prob.get((cur, j), 0.0)
                    - 0.01 * dist[(cur, j)]
                    - 0.001 * max(0.0, -slack)
                )
                candidates.append((score, j, start_service))

            if not candidates:
                break

            candidates.sort(reverse=True)
            _, nxt, start_service = candidates[0]

            route.append((cur, nxt))
            unserved.remove(nxt)
            cur_load += demand[nxt]
            cur_time = start_service
            cur = nxt

        if cur != depot and (cur, depot) in arcs:
            route.append((cur, depot))

        if not route:
            break

        used_arcs.extend(route)
        vehicles_used += 1

    return used_arcs


# ============================================================
# Solver
# ============================================================

def solve_vrptw_pruned(
    instance_path: str,
    vehicle_capacity: float = 200.0,
    max_vehicles: int = 30,
    time_limit: int = 300,
    threads: int = 0,
    mip_gap: float = 0.01,
    lambda_prob: float = 0.05,
    warm_start_prob: float = 0.85,
    strong_keep_prob: float = 0.70,
    soft_keep_prob: float = 0.35,
    top_k_prob: int = 3,
    top_k_dist: int = 3,
    knn_repair_k: int = 4,
    min_degree: int = 3,
    tight_extra_k: int = 2,
    mip_focus: int = 1,
    numeric_focus: int = 1,
    log_to_console: bool = False,
) -> Dict:
    instance_name = os.path.basename(instance_path)
    print(f"\n{'=' * 80}")
    print(f"Solving: {instance_name}")
    print(f"{'=' * 80}")

    try:
        nodes_csv = os.path.join(instance_path, f"{instance_name}_nodes.csv")
        edges_csv = os.path.join(instance_path, f"{instance_name}_edges_solver.csv")

        if not os.path.exists(nodes_csv):
            raise FileNotFoundError(f"Missing nodes file: {nodes_csv}")
        if not os.path.exists(edges_csv):
            raise FileNotFoundError(f"Missing edges file: {edges_csv}")

        nodes_df = pd.read_csv(nodes_csv)
        edges_df = pd.read_csv(edges_csv)

        # Column mapping
        node_id_col = get_col(nodes_df, ["node_id"])
        depot_col = get_col(nodes_df, ["is_depot"])
        x_col = get_col(nodes_df, ["x"])
        y_col = get_col(nodes_df, ["y"])
        demand_col = get_col(nodes_df, ["demand", "DEMAND"])
        ready_col = get_col(nodes_df, ["ready_time", "READY TIME"])
        due_col = get_col(nodes_df, ["due_date", "DUE DATE"])
        service_col = get_col(nodes_df, ["service_time", "SERVICE TIME"])

        depot_rows = nodes_df[nodes_df[depot_col] == 1]
        if len(depot_rows) != 1:
            raise ValueError(f"Expected exactly one depot, found {len(depot_rows)}")

        depot = int(depot_rows.iloc[0][node_id_col])
        V = [int(v) for v in nodes_df[node_id_col].tolist()]
        C = [v for v in V if v != depot]

        nodes_indexed = nodes_df.set_index(node_id_col)
        demand = {int(k): float(v) for k, v in nodes_indexed[demand_col].to_dict().items()}
        a = {int(k): float(v) for k, v in nodes_indexed[ready_col].to_dict().items()}
        b = {int(k): float(v) for k, v in nodes_indexed[due_col].to_dict().items()}
        s = {int(k): float(v) for k, v in nodes_indexed[service_col].to_dict().items()}

        coords = {
            int(r[node_id_col]): (float(r[x_col]), float(r[y_col]))
            for _, r in nodes_df.iterrows()
        }

        full_dist = build_full_distance_dict(nodes_df)

        # Safer repair
        repaired_edges_df = repair_pruned_graph_safer(
            nodes_df=nodes_df,
            edges_df=edges_df,
            depot=depot,
            a=a,
            b=b,
            s=s,
            strong_keep_prob=strong_keep_prob,
            soft_keep_prob=soft_keep_prob,
            top_k_prob=top_k_prob,
            top_k_dist=top_k_dist,
            min_degree=min_degree,
            knn_repair_k=knn_repair_k,
            tight_extra_k=tight_extra_k,
        )

        repaired_path = os.path.join(instance_path, f"{instance_name}_edges_repaired_safe.csv")
        repaired_edges_df.to_csv(repaired_path, index=False)

        # Arc set
        A = []
        dist = {}
        prob = {}

        for _, r in repaired_edges_df.iterrows():
            i, j = int(r["from"]), int(r["to"])
            if i == j:
                continue

            # Final time-feasibility filter before model
            if not is_time_feasible(i, j, a, b, s, full_dist):
                continue

            A.append((i, j))
            dist[(i, j)] = float(r["distance"]) if "distance" in repaired_edges_df.columns else full_dist[(i, j)]
            prob[(i, j)] = float(r["prob"]) if "prob" in repaired_edges_df.columns and pd.notna(r["prob"]) else 0.0

        A = list(dict.fromkeys(A))
        A_set = set(A)

        # Feasibility audit
        audit = audit_graph_feasibility(
            V=V,
            C=C,
            depot=depot,
            arcs=A_set,
            a=a,
            b=b,
            s=s,
            dist=dist
        )

        if not audit["ok"]:
            print("WARNING: Feasibility audit found issues.")
            print(" missing_out      :", audit["missing_out"][:10])
            print(" missing_in       :", audit["missing_in"][:10])
            print(" missing_from_dep :", audit["missing_from_depot"][:10])
            print(" missing_to_dep   :", audit["missing_to_depot"][:10])

        print(f"Nodes: {len(V)} | Customers: {len(C)} | Arcs after safe repair: {len(A)}")

        # Build model
        m = gp.Model(instance_name)

        x = m.addVars(A, vtype=GRB.BINARY, name="x")
        t = m.addVars(V, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
        u = m.addVars(V, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

        # Mild objective shaping
        m.setObjective(
            gp.quicksum(
                (dist[i, j] - lambda_prob * prob.get((i, j), 0.0)) * x[i, j]
                for (i, j) in A
            ),
            GRB.MINIMIZE
        )

        # Time windows
        for i in V:
            m.addConstr(t[i] >= a[i], name=f"tw_lb_{i}")
            m.addConstr(t[i] <= b[i], name=f"tw_ub_{i}")

        # Load bounds
        m.addConstr(u[depot] == 0.0, name="load_depot")
        for i in C:
            m.addConstr(u[i] >= demand[i], name=f"load_lb_{i}")
            m.addConstr(u[i] <= vehicle_capacity, name=f"load_ub_{i}")

        outgoing = {i: [] for i in V}
        incoming = {i: [] for i in V}
        for i, j in A:
            outgoing[i].append((i, j))
            incoming[j].append((i, j))

        # Visit each customer exactly once
        for i in C:
            if len(outgoing[i]) == 0:
                raise ValueError(f"Customer {i} has no outgoing arcs after safe repair")
            if len(incoming[i]) == 0:
                raise ValueError(f"Customer {i} has no incoming arcs after safe repair")

            m.addConstr(
                gp.quicksum(x[arc] for arc in outgoing[i]) == 1,
                name=f"out_{i}"
            )
            m.addConstr(
                gp.quicksum(x[arc] for arc in incoming[i]) == 1,
                name=f"in_{i}"
            )

        # Depot balance + vehicle bounds
        if len(outgoing[depot]) == 0 or len(incoming[depot]) == 0:
            raise ValueError("Depot has no incoming/outgoing arcs after safe repair")

        dep_out = gp.quicksum(x[arc] for arc in outgoing[depot])
        dep_in = gp.quicksum(x[arc] for arc in incoming[depot])

        m.addConstr(dep_out == dep_in, name="depot_balance")
        m.addConstr(dep_out <= max_vehicles, name="vehicle_limit")

        min_vehicles = math.ceil(sum(demand[i] for i in C) / vehicle_capacity)
        m.addConstr(dep_out >= min_vehicles, name="vehicle_lb")

        # Time precedence
        for (i, j) in A:
            if j == depot:
                continue
            M_ij = max(0.0, b[i] + s[i] + dist[i, j] - a[j])
            m.addConstr(
                t[j] >= t[i] + s[i] + dist[i, j] - M_ij * (1 - x[i, j]),
                name=f"time_{i}_{j}"
            )

        # Capacity propagation
        for (i, j) in A:
            if i != depot and j != depot:
                m.addConstr(
                    u[j] >= u[i] + demand[j] - vehicle_capacity * (1 - x[i, j]),
                    name=f"cap_{i}_{j}"
                )

        # Branch priorities
        for (i, j) in A:
            p = max(0.0, min(1.0, prob.get((i, j), 0.0)))
            x[i, j].BranchPriority = int(round(100 * p))

        # Greedy warm start
        warm_arcs = build_greedy_warm_start(
            V=V,
            C=C,
            depot=depot,
            arcs=A_set,
            demand=demand,
            a=a,
            b=b,
            s=s,
            dist=dist,
            prob=prob,
            vehicle_capacity=vehicle_capacity,
            max_vehicles=max_vehicles,
        )
        warm_arc_set = set(warm_arcs)

        for (i, j) in A:
            if (i, j) in warm_arc_set:
                x[i, j].Start = 1.0
            elif prob.get((i, j), 0.0) >= warm_start_prob:
                x[i, j].Start = 1.0
            else:
                x[i, j].Start = 0.0

        # Gurobi params
        m.setParam(GRB.Param.TimeLimit, time_limit)
        m.setParam(GRB.Param.Threads, threads)
        m.setParam(GRB.Param.MIPGap, mip_gap)
        m.setParam(GRB.Param.OutputFlag, 1 if log_to_console else 0)
        m.setParam(GRB.Param.MIPFocus, mip_focus)
        m.setParam(GRB.Param.NumericFocus, numeric_focus)
        m.setParam(GRB.Param.Presolve, 2)
        m.setParam(GRB.Param.Cuts, 2)
        m.setParam(GRB.Param.Heuristics, 0.2)
        m.setParam(GRB.Param.Symmetry, 2)

        m.optimize()

        result = {
            "instance": instance_name,
            "status": int(m.Status),
            "found": False,
            "objective": None,
            "best_bound": None,
            "gap": None,
            "routes": [],
            "num_routes": 0,
            "time": float(m.Runtime),
            "arcs": len(A),
            "customers": len(C),
        }

        if m.Status == GRB.INFEASIBLE:
            print("Status: INFEASIBLE")
            try:
                iis_path = os.path.join(instance_path, f"{instance_name}_iis.ilp")
                m.computeIIS()
                m.write(iis_path)
                result["iis_file"] = iis_path
                print(f"IIS written to: {iis_path}")
            except Exception as e:
                print(f"Could not write IIS: {e}")
            return result

        if m.SolCount > 0 and m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            result["found"] = True
            result["objective"] = float(m.ObjVal)
            try:
                result["best_bound"] = float(m.ObjBound)
            except Exception:
                pass
            try:
                result["gap"] = float(m.MIPGap)
            except Exception:
                pass

            used_arcs = [(i, j) for (i, j) in A if x[i, j].X > 0.5]

            routes = extract_routes(used_arcs, depot)
            result["routes"] = routes
            result["num_routes"] = len(routes)

            # Save used arcs
            sol_edges_rows = []
            for i, j in used_arcs:
                sol_edges_rows.append({
                    "from": i,
                    "to": j,
                    "distance": dist[(i, j)],
                    "prob": prob.get((i, j), 0.0),
                })
            sol_edges_df = pd.DataFrame(sol_edges_rows)
            sol_edges_path = os.path.join(instance_path, f"{instance_name}_solution_edges.csv")
            sol_edges_df.to_csv(sol_edges_path, index=False)

            # Save route summary
            route_rows = []
            for ridx, route in enumerate(routes, 1):
                route_rows.append({
                    "route_id": ridx,
                    "route": " -> ".join(map(str, route)),
                    "num_nodes": len(route),
                    "distance": compute_route_distance(route, dist),
                })
            routes_df = pd.DataFrame(route_rows)
            routes_path = os.path.join(instance_path, f"{instance_name}_routes.csv")
            routes_df.to_csv(routes_path, index=False)

            status_name = {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.SUBOPTIMAL: "SUBOPTIMAL",
            }.get(m.Status, str(m.Status))

            print(f"Objective: {result['objective']}")
            print(f"Routes: {result['num_routes']}")
            for ridx, route in enumerate(routes, 1):
                print(f"Route {ridx}: {' -> '.join(map(str, route))}")

            return result

        print("Status: NO_SOLUTION")
        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            "instance": instance_name,
            "status": -1,
            "found": False,
            "error": str(e),
        }


# ============================================================
# Run all instances
# ============================================================

def main():
    base_path = os.path.join(os.path.dirname(__file__), "safe_pruned_graphs")

    if not os.path.exists(base_path):
        print(f"ERROR: {base_path} not found")
        sys.exit(1)

    instances = sorted(
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("inst_")
    )

    if not instances:
        print(f"ERROR: No instances found in {base_path}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"SOLVING {len(instances)} INSTANCES")
    print(f"{'=' * 80}")

    start_all = time.time()
    results = []

    for idx, inst_name in enumerate(instances, 1):
        print(f"\n[{idx}/{len(instances)}] {inst_name}")
        instance_path = os.path.join(base_path, inst_name)

        result = solve_vrptw_pruned(
            instance_path=instance_path,
            vehicle_capacity=200.0,
            max_vehicles=25,
            time_limit=300,
            threads=0,
            mip_gap=0.01,
            lambda_prob=0.05,
            warm_start_prob=0.85,
            strong_keep_prob=0.70,
            soft_keep_prob=0.35,
            top_k_prob=3,
            top_k_dist=3,
            knn_repair_k=4,
            min_degree=3,
            tight_extra_k=2,
            mip_focus=1,
            numeric_focus=1,
            log_to_console=True,
        )
        results.append(result)

    total_time = time.time() - start_all

    summary_rows = []
    solved = 0

    for r in results:
        ok = r.get("found", False)
        solved += int(ok)
        status_str = "OPTIMAL" if r.get("status") == GRB.OPTIMAL else (
            "TIME_LIMIT" if r.get("status") == GRB.TIME_LIMIT else (
                "INFEASIBLE" if r.get("status") == GRB.INFEASIBLE else (
                    "SUBOPTIMAL" if r.get("status") == GRB.SUBOPTIMAL else "FAIL"
                )
            )
        )
        summary_rows.append({
            "instance": r.get("instance"),
            "status": status_str,
            "objective": f"{r.get('objective'):.1f}" if r.get("found") else "N/A",
            "routes": r.get("num_routes", 0),
            "gap_%": f"{r.get('gap', 0)*100:.2f}" if r.get("found") else "N/A",
            "time_s": f"{r.get('time', 0):.2f}",
            "arcs": r.get("arcs", 0),
            "solved": "YES" if ok else "NO"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(base_path, "solver_summary_safe.csv")
    summary_df.to_csv(summary_file, index=False)

    print(f"\n{'=' * 80}")
    print(f"EXECUTION SUMMARY")
    print(f"{'=' * 80}\n")
    print(f"Total Instances: {len(instances)}")
    print(f"Successfully Solved: {solved}/{len(instances)}")
    print(f"Total Execution Time: {total_time:.1f}s\n")

    print("RESULTS TABLE:")
    print(summary_df.to_string(index=False))
    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {summary_file}\n")


if __name__ == "__main__":
    main()