import os
import glob
import math
import json
import argparse
from typing import Dict, List, Tuple, Set

import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

console = Console()


# ============================================================
# IO helpers
# ============================================================
def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_depot_id_from_nodes(node_df: pd.DataFrame) -> int:
    if "is_depot" in node_df.columns:
        depots = node_df.loc[
            pd.to_numeric(node_df["is_depot"], errors="coerce").fillna(0).astype(int) == 1,
            "node_id"
        ].tolist()
        if len(depots) == 1:
            return int(depots[0])

    node_ids = pd.to_numeric(node_df["node_id"], errors="coerce").dropna().astype(int)
    uniq = set(node_ids.tolist())
    if 0 in uniq:
        return 0
    if 1 in uniq:
        return 1
    return int(node_ids.min())


def normalize_top_pcts(values: List[float]) -> List[float]:
    out = []
    for v in values:
        if v > 1.0:
            v = v / 100.0
        if v <= 0 or v > 1:
            raise ValueError(f"Invalid top percentage: {v}")
        out.append(float(v))
    return out


def detect_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]

    if required:
        raise ValueError(
            f"Missing required column. Expected one of {candidates}. "
            f"Found columns: {list(df.columns)}"
        )
    return ""


def build_solver_nodes_csv(node_df: pd.DataFrame) -> pd.DataFrame:
    node_id_col = detect_col(node_df, ["node_id", "id", "node"])
    demand_col = detect_col(node_df, ["demand"])
    ready_col = detect_col(node_df, ["ready_time", "tw_start", "earliest", "time_window_start"])
    due_col = detect_col(node_df, ["due_time", "due_date", "tw_end", "latest", "time_window_end"])
    service_col = detect_col(node_df, ["service_time", "service", "service_duration"])

    is_depot_col = detect_col(node_df, ["is_depot", "depot"], required=False)
    x_col = detect_col(node_df, ["x", "x_coord", "coord_x"], required=False)
    y_col = detect_col(node_df, ["y", "y_coord", "coord_y"], required=False)

    out = pd.DataFrame()
    out["node_id"] = pd.to_numeric(node_df[node_id_col], errors="coerce")
    out["demand"] = pd.to_numeric(node_df[demand_col], errors="coerce").fillna(0.0)
    out["ready_time"] = pd.to_numeric(node_df[ready_col], errors="coerce").fillna(0.0)
    out["due_time"] = pd.to_numeric(node_df[due_col], errors="coerce").fillna(0.0)
    out["service_time"] = pd.to_numeric(node_df[service_col], errors="coerce").fillna(0.0)

    if is_depot_col:
        out["is_depot"] = pd.to_numeric(node_df[is_depot_col], errors="coerce").fillna(0).astype(int)
    else:
        tmp = pd.DataFrame({"node_id": out["node_id"]}).dropna().copy()
        tmp["node_id"] = tmp["node_id"].astype(int)
        depot_id = detect_depot_id_from_nodes(tmp.assign(is_depot=0))
        out["is_depot"] = (
            pd.to_numeric(out["node_id"], errors="coerce").fillna(-1).astype(int) == depot_id
        ).astype(int)

    if x_col:
        out["x"] = pd.to_numeric(node_df[x_col], errors="coerce")
    if y_col:
        out["y"] = pd.to_numeric(node_df[y_col], errors="coerce")

    out = out.dropna(subset=["node_id"]).copy()
    out["node_id"] = out["node_id"].astype(int)
    out["is_depot"] = pd.to_numeric(out["is_depot"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values("node_id").reset_index(drop=True)
    return out


# ============================================================
# Core graph builder
# ============================================================
def build_score_map(ranked_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    score_map: Dict[Tuple[int, int], float] = {}
    cols = list(ranked_df.columns)
    f_idx, t_idx, s_idx = cols.index("from"), cols.index("to"), cols.index("score")

    for row in ranked_df.itertuples(index=False, name=None):
        u = int(row[f_idx])
        v = int(row[t_idx])
        s = float(row[s_idx])
        score_map[(u, v)] = s

    return score_map


def top_pct_edges(ranked_df: pd.DataFrame, top_pct: float) -> Set[Tuple[int, int]]:
    n = len(ranked_df)
    if n == 0:
        return set()

    k = max(1, int(math.ceil(n * top_pct)))
    top_df = ranked_df.head(k)
    return set((int(r[0]), int(r[1])) for r in top_df[["from", "to"]].itertuples(index=False, name=None))


def get_customer_nodes(node_df: pd.DataFrame, depot_id: int) -> List[int]:
    node_ids = pd.to_numeric(node_df["node_id"], errors="coerce").dropna().astype(int).tolist()
    return [nid for nid in node_ids if nid != depot_id]


def add_depot_edges(
    final_edges: Set[Tuple[int, int]],
    score_map: Dict[Tuple[int, int], float],
    customers: List[int],
    depot_id: int,
) -> int:
    added = 0
    for i in customers:
        if (depot_id, i) in score_map and (depot_id, i) not in final_edges:
            final_edges.add((depot_id, i))
            added += 1
        if (i, depot_id) in score_map and (i, depot_id) not in final_edges:
            final_edges.add((i, depot_id))
            added += 1
    return added


def sorted_outgoing_customer_edges(
    node_id: int,
    customers: Set[int],
    score_map: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int]]:
    cand = []
    for j in customers:
        if j == node_id:
            continue
        e = (node_id, j)
        if e in score_map:
            cand.append(e)
    cand.sort(key=lambda e: score_map[e], reverse=True)
    return cand


def sorted_incoming_customer_edges(
    node_id: int,
    customers: Set[int],
    score_map: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int]]:
    cand = []
    for j in customers:
        if j == node_id:
            continue
        e = (j, node_id)
        if e in score_map:
            cand.append(e)
    cand.sort(key=lambda e: score_map[e], reverse=True)
    return cand


def add_support_edges(
    final_edges: Set[Tuple[int, int]],
    score_map: Dict[Tuple[int, int], float],
    customers: List[int],
    k_out: int,
    k_in: int,
) -> Tuple[int, int]:
    added_out = 0
    added_in = 0
    customer_set = set(customers)

    for i in customers:
        out_edges = sorted_outgoing_customer_edges(i, customer_set, score_map)
        for e in out_edges[:k_out]:
            if e not in final_edges:
                final_edges.add(e)
                added_out += 1

        in_edges = sorted_incoming_customer_edges(i, customer_set, score_map)
        for e in in_edges[:k_in]:
            if e not in final_edges:
                final_edges.add(e)
                added_in += 1

    return added_out, added_in


def remove_self_loops(edges: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    return {e for e in edges if e[0] != e[1]}


def compute_degrees(
    edges: Set[Tuple[int, int]],
    customers: List[int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    in_deg = {i: 0 for i in customers}
    out_deg = {i: 0 for i in customers}

    for u, v in edges:
        if u in out_deg:
            out_deg[u] += 1
        if v in in_deg:
            in_deg[v] += 1

    return in_deg, out_deg


def repair_low_degree_nodes(
    final_edges: Set[Tuple[int, int]],
    score_map: Dict[Tuple[int, int], float],
    customers: List[int],
    min_in_degree: int,
    min_out_degree: int,
) -> Tuple[int, int]:
    added_in = 0
    added_out = 0
    customer_set = set(customers)

    in_deg, out_deg = compute_degrees(final_edges, customers)

    for i in customers:
        if out_deg[i] < min_out_degree:
            cand = sorted_outgoing_customer_edges(i, customer_set, score_map)
            for e in cand:
                if e not in final_edges:
                    final_edges.add(e)
                    added_out += 1
                    out_deg[i] += 1
                    in_deg[e[1]] += 1
                if out_deg[i] >= min_out_degree:
                    break

        if in_deg[i] < min_in_degree:
            cand = sorted_incoming_customer_edges(i, customer_set, score_map)
            for e in cand:
                if e not in final_edges:
                    final_edges.add(e)
                    added_in += 1
                    in_deg[i] += 1
                    out_deg[e[0]] += 1
                if in_deg[i] >= min_in_degree:
                    break

    return added_in, added_out


def build_solver_graph_for_pct(
    ranked_df: pd.DataFrame,
    node_df: pd.DataFrame,
    top_pct: float,
    k_out: int,
    k_in: int,
    min_out_degree: int,
    min_in_degree: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ranked_df = ranked_df.copy()

    required_ranked_cols = {"from", "to", "score"}
    if not required_ranked_cols.issubset(set(ranked_df.columns)):
        raise ValueError(f"Ranked file missing columns: {required_ranked_cols - set(ranked_df.columns)}")

    if "node_id" not in node_df.columns:
        raise ValueError("node_features.csv missing column 'node_id'")

    ranked_df["from"] = pd.to_numeric(ranked_df["from"], errors="coerce").astype("Int64")
    ranked_df["to"] = pd.to_numeric(ranked_df["to"], errors="coerce").astype("Int64")
    ranked_df["score"] = pd.to_numeric(ranked_df["score"], errors="coerce")
    ranked_df = ranked_df.dropna(subset=["from", "to", "score"]).copy()

    ranked_df["from"] = ranked_df["from"].astype(int)
    ranked_df["to"] = ranked_df["to"].astype(int)
    ranked_df = ranked_df.sort_values("score", ascending=False).reset_index(drop=True)

    depot_id = detect_depot_id_from_nodes(node_df)
    customers = get_customer_nodes(node_df, depot_id)
    node_set = set(pd.to_numeric(node_df["node_id"], errors="coerce").dropna().astype(int).tolist())

    ranked_df = ranked_df[
        ranked_df["from"].isin(node_set)
        & ranked_df["to"].isin(node_set)
        & (ranked_df["from"] != ranked_df["to"])
    ].reset_index(drop=True)

    score_map = build_score_map(ranked_df)

    final_edges = top_pct_edges(ranked_df, top_pct)
    top_edge_count = len(final_edges)

    depot_added = add_depot_edges(final_edges, score_map, customers, depot_id)
    support_out_added, support_in_added = add_support_edges(
        final_edges=final_edges,
        score_map=score_map,
        customers=customers,
        k_out=k_out,
        k_in=k_in,
    )

    final_edges = remove_self_loops(final_edges)

    repair_in_added, repair_out_added = repair_low_degree_nodes(
        final_edges=final_edges,
        score_map=score_map,
        customers=customers,
        min_in_degree=min_in_degree,
        min_out_degree=min_out_degree,
    )

    in_deg, out_deg = compute_degrees(final_edges, customers)
    isolated = [i for i in customers if in_deg[i] == 0 or out_deg[i] == 0]
    weak_nodes = [i for i in customers if in_deg[i] < min_in_degree or out_deg[i] < min_out_degree]

    id_col = detect_col(node_df, ["node_id", "id", "node"])
    x_col = detect_col(node_df, ["x", "x_coord", "coord_x"])
    y_col = detect_col(node_df, ["y", "y_coord", "coord_y"])

    x_map = dict(zip(node_df[id_col].astype(int), node_df[x_col].astype(float)))
    y_map = dict(zip(node_df[id_col].astype(int), node_df[y_col].astype(float)))

    records = []
    for u, v in sorted(final_edges):
        dist = math.hypot(x_map[v] - x_map[u], y_map[v] - y_map[u])
        records.append({
            "from": int(u),
            "to": int(v),
            "distance": float(dist),
            "score": float(score_map.get((u, v), -1.0)),
            "selected": 1,
            "top_pct": float(top_pct),
        })

    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df = out_df.sort_values(["score", "from", "to"], ascending=[False, True, True]).reset_index(drop=True)
        out_df["rank_in_solver_graph"] = range(1, len(out_df) + 1)

    metadata = {
        "depot_id": int(depot_id),
        "nodes": int(len(node_set)),
        "customers": int(len(customers)),
        "candidate_edges": int(len(ranked_df)),
        "top_pct": float(top_pct),
        "top_edges_initial": int(top_edge_count),
        "depot_edges_added": int(depot_added),
        "support_out_edges_added": int(support_out_added),
        "support_in_edges_added": int(support_in_added),
        "repair_in_edges_added": int(repair_in_added),
        "repair_out_edges_added": int(repair_out_added),
        "final_edges": int(len(final_edges)),
        "min_in_degree_target": int(min_in_degree),
        "min_out_degree_target": int(min_out_degree),
        "actual_min_in_degree": int(min(in_deg.values())) if in_deg else 0,
        "actual_min_out_degree": int(min(out_deg.values())) if out_deg else 0,
        "isolated_customer_count": int(len(isolated)),
        "weak_customer_count": int(len(weak_nodes)),
        "isolated_customers": [int(x) for x in isolated],
        "weak_customers": [int(x) for x in weak_nodes],
        "k_in": int(k_in),
        "k_out": int(k_out),
    }

    return out_df, metadata


# ============================================================
# File discovery
# ============================================================
def find_ranked_files(ranked_root: str) -> Dict[str, str]:
    """
    Returns:
        {inst_name: ranked_csv_path}

    Supports:
      ranked_root/train/inst_001.csv
      ranked_root/val/inst_002.csv
      ranked_root/test/inst_003.csv
      ranked_root/inst_004.csv
    """
    files = glob.glob(os.path.join(ranked_root, "**", "*.csv"), recursive=True)
    out: Dict[str, str] = {}

    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        if name in out:
            raise ValueError(f"Duplicate ranked file instance name found: {name}")
        out[name] = fp

    return out


def find_node_files(processed_root: str) -> Dict[str, str]:
    """
    Returns:
        {inst_name: node_features_csv_path}

    Supports:
      processed_root/train/inst_001/node_features.csv
      processed_root/val/inst_002/node_features.csv
      processed_root/test/inst_003/node_features.csv
      processed_root/inst_004/node_features.csv
    """
    files = glob.glob(os.path.join(processed_root, "**", "node_features.csv"), recursive=True)
    out: Dict[str, str] = {}

    for fp in files:
        rel = os.path.relpath(fp, processed_root)
        parts = rel.split(os.sep)

        if len(parts) >= 3 and parts[0] in {"train", "val", "test"}:
            inst_name = parts[1]
        elif len(parts) >= 2:
            inst_name = parts[0]
        else:
            continue

        if inst_name in out:
            raise ValueError(f"Duplicate node_features instance name found: {inst_name}")
        out[inst_name] = fp

    return out


# ============================================================
# Output helpers
# ============================================================
def get_instance_output_dir(output_root: str, inst_name: str) -> str:
    return os.path.join(output_root, inst_name)


def pct_tag_from_float(p: float) -> str:
    return f"{int(round(p * 100))}"


# ============================================================
# Batch runner
# ============================================================
def process_instance(
    inst_name: str,
    ranked_file: str,
    node_file: str,
    output_root: str,
    top_pcts: List[float],
    k_out: int,
    k_in: int,
    min_out_degree: int,
    min_in_degree: int,
) -> List[Dict[str, object]]:
    ranked_df = safe_read_csv(ranked_file)
    node_df = safe_read_csv(node_file)

    inst_out_root = get_instance_output_dir(output_root, inst_name)
    os.makedirs(inst_out_root, exist_ok=True)

    solver_nodes_df = build_solver_nodes_csv(node_df)
    nodes_out = os.path.join(inst_out_root, "nodes.csv")
    solver_nodes_df.to_csv(nodes_out, index=False)

    summaries = []
    for p in top_pcts:
        pct_tag = pct_tag_from_float(p)

        solver_df, meta = build_solver_graph_for_pct(
            ranked_df=ranked_df,
            node_df=node_df,
            top_pct=p,
            k_out=k_out,
            k_in=k_in,
            min_out_degree=min_out_degree,
            min_in_degree=min_in_degree,
        )

        edges_out = os.path.join(inst_out_root, f"solver_edges_top_{pct_tag}.csv")
        meta_out = os.path.join(inst_out_root, f"solver_stats_top_{pct_tag}.json")

        solver_df.to_csv(edges_out, index=False)
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        summary = {
            "instance": inst_name,
            "pct": int(round(p * 100)),
            "nodes_csv": nodes_out,
            "edges_csv": edges_out,
            "stats_json": meta_out,
            **meta,
        }
        summaries.append(summary)

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build flat solver graph folders from ranked edges")
    parser.add_argument("--processed_root", type=str, default=None, help="Root containing inst_*/node_features.csv")
    parser.add_argument("--ranked_root", type=str, default=None, help="Root containing ranked edge csv files")
    parser.add_argument("--output_root", type=str, default=None, help="Output root for solver-ready graphs")
    parser.add_argument("--top_pcts", type=float, nargs="+", default=[15, 20, 25, 30, 40, 50])
    parser.add_argument("--k_out", type=int, default=3)
    parser.add_argument("--k_in", type=int, default=3)
    parser.add_argument("--min_out_degree", type=int, default=2)
    parser.add_argument("--min_in_degree", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    processed_root = args.processed_root or os.path.join(script_dir, "processed_data")
    ranked_root = args.ranked_root or os.path.join(script_dir, "ranked_output")
    output_root = args.output_root or os.path.join(script_dir, "solver_graph")

    top_pcts = normalize_top_pcts(args.top_pcts)
    os.makedirs(output_root, exist_ok=True)

    ranked_map = find_ranked_files(ranked_root)
    node_map = find_node_files(processed_root)

    matched_instances = sorted(set(ranked_map.keys()) & set(node_map.keys()))
    if not matched_instances:
        print("ranked instances sample:", list(sorted(ranked_map.keys()))[:10])
        print("node instances sample  :", list(sorted(node_map.keys()))[:10])
        raise FileNotFoundError("No matching instances found between ranked_root and processed_root")

    console.print(Panel("[bold cyan]BUILDING FLAT SOLVER GRAPH[/bold cyan]", box=box.DOUBLE))
    console.print(f"Matched instances: [yellow]{len(matched_instances)}[/yellow]")

    all_summaries: List[Dict[str, object]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing instances...", total=len(matched_instances))

        for inst_name in matched_instances:
            ranked_file = ranked_map[inst_name]
            node_file = node_map[inst_name]

            progress.update(task, description=f"[cyan]Processing {inst_name}...")
            
            summaries = process_instance(
                inst_name=inst_name,
                ranked_file=ranked_file,
                node_file=node_file,
                output_root=output_root,
                top_pcts=top_pcts,
                k_out=args.k_out,
                k_in=args.k_in,
                min_out_degree=args.min_out_degree,
                min_in_degree=args.min_in_degree,
            )
            all_summaries.extend(summaries)
            progress.advance(task)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_csv = os.path.join(output_root, "solver_graph_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        console.print(f"\n[bold green]Done.[/bold green] Saved summary: [cyan]{summary_csv}[/cyan]")
        
        # Build summary table for last processed instances (limit to 20 for readability)
        table = Table(title="Solver Graph Generation Summary (Top 20 results)", box=box.ROUNDED)
        table.add_column("Instance", style="cyan")
        table.add_column("Pct (%)", justify="right")
        table.add_column("Nodes", justify="right")
        table.add_column("Edges", justify="right")
        table.add_column("Isolated", justify="right", style="red")
        table.add_column("Weak", justify="right", style="yellow")

        for s in all_summaries[:20]:
            table.add_row(
                str(s["instance"]),
                str(s["pct"]),
                str(s["nodes"]),
                str(s["final_edges"]),
                str(s["isolated_customer_count"]),
                str(s["weak_customer_count"])
            )
        console.print(table)
        if len(all_summaries) > 20:
            console.print(f"... and [yellow]{len(all_summaries) - 20}[/yellow] more rows.")


if __name__ == "__main__":
    main()