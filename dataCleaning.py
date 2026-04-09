import os
import re
import glob
import json
import math
import argparse
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# Helpers
# =============================================================================
def safe_read_table(file_path: str) -> pd.DataFrame:
    """
    Read CSV first; if that fails, read whitespace-separated text.
    """
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.read_csv(file_path, sep=r"\s+", engine="python")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common Solomon-style column names.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "CUST NO.": "node_id",
        "CUST_NO": "node_id",
        "customer": "node_id",
        "XCOORD.": "x",
        "XCOORD": "x",
        "YCOORD.": "y",
        "YCOORD": "y",
        "DEMAND": "demand",
        "READY TIME": "ready_time",
        "READY_TIME": "ready_time",
        "DUE DATE": "due_date",
        "DUE_DATE": "due_date",
        "SERVICE TIME": "service_time",
        "SERVICE_TIME": "service_time",
    }
    return df.rename(columns=rename_map)


def detect_depot_id(df: pd.DataFrame) -> int:
    """
    Robust depot detection.

    Priority:
    1. node_id == 0
    2. node_id == 1
    3. smallest node_id
    """
    if "node_id" not in df.columns:
        raise ValueError("Missing required column 'node_id'")

    node_ids = pd.to_numeric(df["node_id"], errors="coerce").dropna().astype(int)
    if len(node_ids) == 0:
        raise ValueError("No valid node IDs found")

    uniq = set(node_ids.tolist())
    if 0 in uniq:
        return 0
    if 1 in uniq:
        return 1
    return int(node_ids.min())


def parse_solution_routes(solution_file: str) -> List[List[int]]:
    """
    Reads lines like:
    Route 1 : 83 45 61 84
    Route 2 : 10 20 30
    """
    routes: List[List[int]] = []

    with open(solution_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or not line.lower().startswith("route"):
                continue
            if ":" not in line:
                print(f"  Warning: malformed route line skipped -> {line}")
                continue

            rhs = line.split(":", 1)[1].strip()
            if rhs == "":
                routes.append([])
                continue

            tokens = re.findall(r"-?\d+", rhs)
            if not tokens:
                print(f"  Warning: no integer tokens found in route line skipped -> {line}")
                continue

            try:
                routes.append([int(x) for x in tokens])
            except ValueError:
                print(f"  Warning: non-integer token in route line skipped -> {line}")

    return routes


def build_used_edges(
    routes: List[List[int]],
    depot_id: int,
    valid_node_ids: Set[int],
) -> Tuple[Set[Tuple[int, int]], Set[int], Set[Tuple[int, int]]]:
    """
    Convert route list into directed labeled edges.
    """
    used_edges: Set[Tuple[int, int]] = set()
    invalid_nodes: Set[int] = set()
    self_loops: Set[Tuple[int, int]] = set()

    for r_idx, route in enumerate(routes, start=1):
        if not route:
            continue

        bad = [n for n in route if n not in valid_node_ids]
        if bad:
            invalid_nodes.update(bad)
            print(f"  Warning: Route {r_idx} contains invalid node(s), skipped.")
            continue

        arc = (depot_id, route[0])
        if arc[0] != arc[1]:
            used_edges.add(arc)
        else:
            self_loops.add(arc)

        for i in range(len(route) - 1):
            arc = (route[i], route[i + 1])
            if arc[0] != arc[1]:
                used_edges.add(arc)
            else:
                self_loops.add(arc)

        arc = (route[-1], depot_id)
        if arc[0] != arc[1]:
            used_edges.add(arc)
        else:
            self_loops.add(arc)

    return used_edges, invalid_nodes, self_loops


def ensure_torch_available() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is not installed. Install torch to enable .pt export, or set SAVE_PT = False."
        )


def show_directory_tree(path: str, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
    if current_depth >= max_depth:
        return

    try:
        items = sorted(os.listdir(path))
        dirs = [i for i in items if os.path.isdir(os.path.join(path, i))]
        files = [i for i in items if os.path.isfile(os.path.join(path, i))]

        for i, d in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            print(f"{prefix}{'└── ' if is_last_dir else '├── '}{d}/")
            new_prefix = prefix + ("    " if is_last_dir else "│   ")
            show_directory_tree(
                os.path.join(path, d),
                prefix=new_prefix,
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )

        for i, f in enumerate(files):
            is_last = i == len(files) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{f}")
    except PermissionError:
        pass


# =============================================================================
# Core preprocessing
# =============================================================================
def prepare_node_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Clean, validate, normalize, and reorder nodes so the depot is first.
    """
    df = df.copy()
    required_cols = ["node_id", "x", "y", "ready_time", "due_date", "service_time"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    if "demand" not in df.columns:
        df["demand"] = 0.0

    numeric_cols = ["node_id", "x", "y", "demand", "ready_time", "due_date", "service_time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid nodes found after cleaning")

    df["node_id"] = df["node_id"].astype(int)

    if df["node_id"].duplicated().any():
        dup_ids = sorted(df.loc[df["node_id"].duplicated(), "node_id"].astype(int).tolist())
        raise ValueError(f"Duplicate node IDs detected: {dup_ids[:10]}")

    depot_id = detect_depot_id(df)

    depot_df = df[df["node_id"] == depot_id].copy()
    other_df = df[df["node_id"] != depot_id].copy().sort_values("node_id")
    df = pd.concat([depot_df, other_df], ignore_index=True)

    if int(df.loc[0, "node_id"]) != depot_id:
        raise ValueError("Depot reordering failed")

    df["is_depot"] = (df["node_id"] == depot_id).astype(np.int8)

    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    demand = df["demand"].to_numpy(dtype=np.float32)
    ready = df["ready_time"].to_numpy(dtype=np.float32)
    due = df["due_date"].to_numpy(dtype=np.float32)
    service = df["service_time"].to_numpy(dtype=np.float32)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)

    demand_max = max(float(np.max(demand)), 1.0)
    ready_max = max(float(np.max(ready)), 1.0)
    due_max = max(float(np.max(due)), 1.0)
    service_max = max(float(np.max(service)), 1.0)

    window_width = due - ready
    window_width_max = max(float(np.max(window_width)), 1.0)

    depot_x, depot_y = float(x[0]), float(y[0])
    depot_dist = np.hypot(x - depot_x, y - depot_y).astype(np.float32)
    depot_dist_max = max(float(np.max(depot_dist)), 1.0)

    df["x_norm"] = ((df["x"] - x_min) / x_range).astype(np.float32)
    df["y_norm"] = ((df["y"] - y_min) / y_range).astype(np.float32)
    df["demand_norm"] = (df["demand"] / demand_max).astype(np.float32)
    df["ready_time_norm"] = (df["ready_time"] / due_max).astype(np.float32)
    df["due_date_norm"] = (df["due_date"] / due_max).astype(np.float32)
    df["service_time_norm"] = (df["service_time"] / service_max).astype(np.float32)
    df["window_width"] = window_width.astype(np.float32)
    df["window_width_norm"] = (df["window_width"] / window_width_max).astype(np.float32)
    df["depot_dist"] = depot_dist
    df["depot_dist_norm"] = (depot_dist / depot_dist_max).astype(np.float32)

    stats = {
        "depot_id": int(depot_id),
        "depot_row_index": 0,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_range": float(x_range),
        "y_range": float(y_range),
        "demand_max": float(demand_max),
        "ready_max": float(ready_max),
        "due_max": float(due_max),
        "service_max": float(service_max),
        "window_width_max": float(window_width_max),
        "depot_dist_max": float(depot_dist_max),
    }
    return df, stats


def build_edge_data_vectorized(
    node_df: pd.DataFrame,
    used_edges: Set[Tuple[int, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Build full directed non-self graph with VRPTW-oriented edge features.

    Notes:
    - Keeps the full graph for the original pipeline.
    - Uses mostly normalized features for training stability.
    - is_time_feasible is only a heuristic pairwise signal, not full route feasibility.
    """
    node_ids = node_df["node_id"].to_numpy(dtype=np.int64)
    x = node_df["x"].to_numpy(dtype=np.float32)
    y = node_df["y"].to_numpy(dtype=np.float32)
    demand = node_df["demand"].to_numpy(dtype=np.float32)
    ready = node_df["ready_time"].to_numpy(dtype=np.float32)
    due = node_df["due_date"].to_numpy(dtype=np.float32)
    service = node_df["service_time"].to_numpy(dtype=np.float32)
    is_depot = node_df["is_depot"].to_numpy(dtype=np.int8)

    x_norm = node_df["x_norm"].to_numpy(dtype=np.float32)
    y_norm = node_df["y_norm"].to_numpy(dtype=np.float32)
    demand_norm = node_df["demand_norm"].to_numpy(dtype=np.float32)
    ready_norm = node_df["ready_time_norm"].to_numpy(dtype=np.float32)
    due_norm = node_df["due_date_norm"].to_numpy(dtype=np.float32)
    service_norm = node_df["service_time_norm"].to_numpy(dtype=np.float32)
    window_width_norm = node_df["window_width_norm"].to_numpy(dtype=np.float32)
    depot_dist_norm_node = node_df["depot_dist_norm"].to_numpy(dtype=np.float32)

    N = len(node_ids)
    if N < 2:
        raise ValueError("Need at least 2 nodes to build directed edges")

    row_idx, col_idx = np.indices((N, N))
    mask = row_idx != col_idx
    src = row_idx[mask].astype(np.int64)
    dst = col_idx[mask].astype(np.int64)

    from_node = node_ids[src]
    to_node = node_ids[dst]

    dx = (x[dst] - x[src]).astype(np.float32)
    dy = (y[dst] - y[src]).astype(np.float32)
    abs_dx = np.abs(dx).astype(np.float32)
    abs_dy = np.abs(dy).astype(np.float32)
    distance = np.hypot(dx, dy).astype(np.float32)
    travel_time = distance.copy()

    due_max = max(float(np.max(due)), 1.0)
    coord_scale = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)

    earliest_departure = (ready[src] + service[src]).astype(np.float32)
    earliest_arrival = (earliest_departure + travel_time).astype(np.float32)
    waiting_time = np.maximum(0.0, ready[dst] - earliest_arrival).astype(np.float32)
    tw_slack = (due[dst] - earliest_arrival).astype(np.float32)
    slack_after_service = (due[dst] - (earliest_arrival + service[dst])).astype(np.float32)
    is_time_feasible = (earliest_arrival <= due[dst]).astype(np.int8)

    from_is_depot = is_depot[src].astype(np.int8)
    to_is_depot = is_depot[dst].astype(np.int8)
    touches_depot = ((from_is_depot + to_is_depot) > 0).astype(np.int8)

    depot_dist_from_norm = depot_dist_norm_node[src].astype(np.float32)
    depot_dist_to_norm = depot_dist_norm_node[dst].astype(np.float32)
    relative_depot_gap_norm = (depot_dist_to_norm - depot_dist_from_norm).astype(np.float32)

    tw_overlap = np.maximum(
        0.0,
        np.minimum(due[src], due[dst]) - np.maximum(ready[src], ready[dst]),
    ).astype(np.float32)
    tw_overlap_norm = (tw_overlap / due_max).astype(np.float32)

    route_proxy_cost = (distance + depot_dist_to_norm * coord_scale).astype(np.float32)

    distance_norm = (distance / coord_scale).astype(np.float32)
    waiting_time_norm = (waiting_time / due_max).astype(np.float32)
    tw_slack_norm = (tw_slack / due_max).astype(np.float32)
    slack_after_service_norm = (slack_after_service / due_max).astype(np.float32)
    arrival_to_norm = (earliest_arrival / due_max).astype(np.float32)
    route_proxy_cost_norm = (route_proxy_cost / max(2.0 * coord_scale, 1.0)).astype(np.float32)

    demand_gap_norm = (demand_norm[dst] - demand_norm[src]).astype(np.float32)
    ready_gap_norm = ((ready[dst] - ready[src]) / due_max).astype(np.float32)
    due_gap_norm = ((due[dst] - due[src]) / due_max).astype(np.float32)
    service_gap_norm = ((service[dst] - service[src]) / max(float(np.max(service)), 1.0)).astype(np.float32)

    edge_features_df = pd.DataFrame({
        "from": from_node,
        "to": to_node,

        # geometric
        "distance_norm": distance_norm,
        "dx": dx,
        "dy": dy,
        "abs_dx": abs_dx,
        "abs_dy": abs_dy,

        # node-context-on-edge
        "x_from_norm": x_norm[src],
        "y_from_norm": y_norm[src],
        "x_to_norm": x_norm[dst],
        "y_to_norm": y_norm[dst],
        "demand_from_norm": demand_norm[src],
        "demand_to_norm": demand_norm[dst],
        "demand_gap_norm": demand_gap_norm,
        "ready_from_norm": ready_norm[src],
        "ready_to_norm": ready_norm[dst],
        "due_from_norm": due_norm[src],
        "due_to_norm": due_norm[dst],
        "ready_gap_norm": ready_gap_norm,
        "due_gap_norm": due_gap_norm,
        "service_from_norm": service_norm[src],
        "service_to_norm": service_norm[dst],
        "service_gap_norm": service_gap_norm,
        "window_width_from_norm": window_width_norm[src],
        "window_width_to_norm": window_width_norm[dst],

        # time-window / routing heuristics
        "arrival_to_norm": arrival_to_norm,
        "waiting_time_norm": waiting_time_norm,
        "tw_slack_norm": tw_slack_norm,
        "slack_after_service_norm": slack_after_service_norm,
        "is_time_feasible": is_time_feasible,
        "tw_overlap_norm": tw_overlap_norm,

        # depot context
        "from_is_depot": from_is_depot,
        "to_is_depot": to_is_depot,
        "touches_depot": touches_depot,
        "depot_dist_from_norm": depot_dist_from_norm,
        "depot_dist_to_norm": depot_dist_to_norm,
        "relative_depot_gap_norm": relative_depot_gap_norm,
        "route_proxy_cost_norm": route_proxy_cost_norm,
    })

    label_set = set(used_edges)
    labels = np.array(
        [1 if (int(f), int(t)) in label_set else 0 for f, t in zip(from_node, to_node)],
        dtype=np.int8,
    )

    edge_index_df = edge_features_df[["from", "to"]].copy()
    y_label_df = edge_index_df.copy()
    y_label_df["label"] = labels

    metadata = {
        "candidate_edges": int(len(edge_features_df)),
        "time_feasible_ratio": float(np.mean(is_time_feasible)),
    }
    return edge_features_df, edge_index_df, y_label_df, metadata


def build_pt_data(
    node_df: pd.DataFrame,
    edge_features_df: pd.DataFrame,
    y_label_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Build compact PyTorch tensors for GNN training.
    """
    ensure_torch_available()

    node_feature_cols = [
        "x_norm",
        "y_norm",
        "demand_norm",
        "ready_time_norm",
        "due_date_norm",
        "service_time_norm",
        "window_width_norm",
        "depot_dist_norm",
        "is_depot",
    ]

    edge_feature_cols = [
        "distance_norm",
        "dx",
        "dy",
        "abs_dx",
        "abs_dy",
        "x_from_norm",
        "y_from_norm",
        "x_to_norm",
        "y_to_norm",
        "demand_from_norm",
        "demand_to_norm",
        "demand_gap_norm",
        "ready_from_norm",
        "ready_to_norm",
        "due_from_norm",
        "due_to_norm",
        "ready_gap_norm",
        "due_gap_norm",
        "service_from_norm",
        "service_to_norm",
        "service_gap_norm",
        "window_width_from_norm",
        "window_width_to_norm",
        "arrival_to_norm",
        "waiting_time_norm",
        "tw_slack_norm",
        "slack_after_service_norm",
        "is_time_feasible",
        "tw_overlap_norm",
        "from_is_depot",
        "to_is_depot",
        "touches_depot",
        "depot_dist_from_norm",
        "depot_dist_to_norm",
        "relative_depot_gap_norm",
        "route_proxy_cost_norm",
    ]

    node_ids = node_df["node_id"].tolist()
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    src_idx = edge_features_df["from"].map(node_id_to_idx).to_numpy(dtype=np.int64)
    dst_idx = edge_features_df["to"].map(node_id_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.vstack([src_idx, dst_idx])

    x_tensor = torch.tensor(node_df[node_feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_features_df[edge_feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(y_label_df["label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    return {
        "x": x_tensor,
        "edge_index": edge_index_tensor,
        "edge_attr": edge_attr_tensor,
        "y": y_tensor,
        "node_ids": torch.tensor(node_ids, dtype=torch.long),
        "node_feature_cols": node_feature_cols,
        "edge_feature_cols": edge_feature_cols,
    }


def preprocess_instance(
    data_file: str,
    solution_file: str,
    output_dir: str,
    save_pt: bool = True,
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 88}")
    print(f"Processing: {data_file}")
    print(f"Solution  : {solution_file}")
    print(f"Output    : {output_dir}")
    print(f"{'=' * 88}")

    raw_df = safe_read_table(data_file)
    raw_df = clean_columns(raw_df)
    node_df, stats = prepare_node_dataframe(raw_df)

    depot_id = int(stats["depot_id"])
    node_ids = node_df["node_id"].tolist()
    node_set = set(node_ids)
    N = len(node_ids)

    print(f"  Detected depot_id: {depot_id}")
    print(f"  First 5 node IDs : {node_ids[:5]}")

    routes = parse_solution_routes(solution_file)
    print(f"  Nodes: {N}, Routes: {len(routes)}")

    used_edges, invalid_nodes, self_loops = build_used_edges(routes, depot_id, node_set)
    if invalid_nodes:
        print(f"  Warning: invalid route node IDs found: {sorted(invalid_nodes)}")
    if self_loops:
        print(f"  Warning: self-loop solution edges ignored: {sorted(self_loops)}")

    edge_features_df, edge_index_df, y_label_df, edge_meta = build_edge_data_vectorized(node_df, used_edges)

    all_generated_edges = set(zip(edge_index_df["from"], edge_index_df["to"]))
    preserved_edges = used_edges & all_generated_edges
    missing_solution_edges = sorted(list(used_edges - all_generated_edges))

    if missing_solution_edges:
        print(f"  Warning: missing solution edges detected: {missing_solution_edges}")
    else:
        print("  ✓ All solution edges preserved in generated graph")

    pos_edges = int(y_label_df["label"].sum())
    total_edges = int(len(y_label_df))
    neg_edges = total_edges - pos_edges
    pos_ratio = float(pos_edges / max(total_edges, 1))
    pos_weight = float(neg_edges / max(pos_edges, 1))

    stats_summary = {
        "instance": os.path.basename(output_dir),
        "depot_id": depot_id,
        "nodes": int(N),
        "routes": int(len(routes)),
        "edges": total_edges,
        "solution_edges": int(len(used_edges)),
        "preserved_edges": int(len(preserved_edges)),
        "missing_solution_edges": int(len(missing_solution_edges)),
        "time_feasible_ratio": float(edge_meta["time_feasible_ratio"]),
        "positive_edges": pos_edges,
        "negative_edges": neg_edges,
        "positive_ratio": pos_ratio,
        "pos_weight": pos_weight,
        "node_feature_dim": 9,
        "edge_feature_dim": 36,
    }

    node_out = os.path.join(output_dir, "node_features.csv")
    edge_index_out = os.path.join(output_dir, "edge_index.csv")
    edge_feat_out = os.path.join(output_dir, "edge_features.csv")
    label_out = os.path.join(output_dir, "y_label.csv")
    stats_out = os.path.join(output_dir, "stats.json")

    node_df.to_csv(node_out, index=False)
    edge_index_df.to_csv(edge_index_out, index=False)
    edge_features_df.to_csv(edge_feat_out, index=False)
    y_label_df.to_csv(label_out, index=False)

    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    if save_pt:
        pt_out = os.path.join(output_dir, "graph_data.pt")
        pt_data = build_pt_data(node_df, edge_features_df, y_label_df)
        torch.save(pt_data, pt_out)
        print(f"  ✓ Saved PyTorch graph data: {pt_out}")

    print(f"  Total edges         : {total_edges:,}")
    print(f"  Solution edges      : {len(used_edges)}")
    print(f"  Preserved edges     : {len(preserved_edges)}")
    print(f"  Positive ratio      : {pos_ratio:.6f}")
    print(f"  Pos weight          : {pos_weight:.4f}")
    print(f"  Time-feasible ratio : {edge_meta['time_feasible_ratio']:.6f}")
    print(f"  ✓ Files saved to {output_dir}")

    return stats_summary


# =============================================================================
# Batch runner
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refined VRPTW preprocessing for GNN edge ranking")
    parser.add_argument("--input_root", type=str, default=None, help="Root directory containing inst_* folders")
    parser.add_argument("--output_root", type=str, default=None, help="Output directory for processed data")
    parser.add_argument("--save_pt", action="store_true", help="Save graph_data.pt")
    parser.add_argument("--no_save_pt", action="store_true", help="Disable graph_data.pt saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = args.input_root or os.path.join(script_dir, "dataset")
    output_root = args.output_root or os.path.join(script_dir, "processed_data")

    save_pt = True
    if args.no_save_pt:
        save_pt = False
    elif args.save_pt:
        save_pt = True

    if not os.path.isdir(input_root):
        print(f"ERROR: Input directory not found at {input_root}")
        raise SystemExit(1)

    instance_dirs = sorted(
        d for d in glob.glob(os.path.join(input_root, "inst_*")) if os.path.isdir(d)
    )

    print(f"Found {len(instance_dirs)} instance folders.\n")
    print(f"{'Instance':<15} {'Data File':<15} {'Solution File':<15} {'Status':<20}")
    print("-" * 75)

    valid_pairs = []
    for inst_dir in instance_dirs:
        inst_name = os.path.basename(inst_dir)
        data_file = os.path.join(inst_dir, "data.csv")
        solution_file = os.path.join(inst_dir, "solution.txt")

        if os.path.exists(data_file) and os.path.exists(solution_file):
            print(f"{inst_name:<15} {'data.csv':<15} {'solution.txt':<15} {'✓ Found':<20}")
            valid_pairs.append((inst_name, data_file, solution_file))
        else:
            missing = []
            if not os.path.exists(data_file):
                missing.append("data.csv")
            if not os.path.exists(solution_file):
                missing.append("solution.txt")
            print(f"{inst_name:<15} {'data.csv':<15} {'solution.txt':<15} {('Missing: ' + ', '.join(missing)):<20}")

    print("-" * 75)
    print(f"\nValid instances: {len(valid_pairs)}/{len(instance_dirs)}")

    if len(valid_pairs) == 0:
        print("No valid instance folders found.")
        raise SystemExit(1)

    if save_pt and torch is None:
        print("ERROR: save_pt=True but torch is not installed.")
        print("Install torch, or run with --no_save_pt.")
        raise SystemExit(1)

    os.makedirs(output_root, exist_ok=True)

    results = []
    for inst_name, data_file, solution_file in valid_pairs:
        output_dir = os.path.join(output_root, inst_name)
        result = preprocess_instance(
            data_file=data_file,
            solution_file=solution_file,
            output_dir=output_dir,
            save_pt=save_pt,
        )
        results.append(result)

    print(f"\n{'=' * 126}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 126}")
    print(
        f"{'Instance':<15} {'Nodes':<8} {'Routes':<8} {'Edges':<10} {'Pos':<8} "
        f"{'Neg':<10} {'PosRatio':<12} {'PosWeight':<12} {'TF_Ratio':<12} {'Missing':<10}"
    )
    print("-" * 126)

    for r in results:
        print(
            f"{r['instance']:<15}"
            f"{r['nodes']:<8}"
            f"{r['routes']:<8}"
            f"{r['edges']:<10}"
            f"{r['positive_edges']:<8}"
            f"{r['negative_edges']:<10}"
            f"{r['positive_ratio']:<12.6f}"
            f"{r['pos_weight']:<12.4f}"
            f"{r['time_feasible_ratio']:<12.6f}"
            f"{r['missing_solution_edges']:<10}"
        )

    print("-" * 126)
    print(f"\nOutput directory structure:")
    print(f"{output_root}/")
    show_directory_tree(output_root)


if __name__ == "__main__":
    main()
