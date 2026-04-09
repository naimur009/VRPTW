import os
import json
import math
import time
import argparse
from typing import Dict, List, Optional

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console, Group
from rich.rule import Rule

console = Console()


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise ValueError(
            f"Missing required column. Expected one of {candidates}. Found columns: {list(df.columns)}"
        )
    return None


# ============================================================
# File discovery
# ============================================================
def discover_instances(solver_graph_root: str, instances: Optional[List[str]] = None):
    found = []

    for inst_name in sorted(os.listdir(solver_graph_root)):
        inst_dir = os.path.join(solver_graph_root, inst_name)

        if not os.path.isdir(inst_dir):
            continue

        if instances is not None and len(instances) > 0 and inst_name not in instances:
            continue

        if not os.path.isfile(os.path.join(inst_dir, "nodes.csv")):
            continue

        found.append((inst_name, inst_dir))

    return found


def find_nodes_file(inst_dir: str) -> str:
    path = os.path.join(inst_dir, "nodes.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"nodes.csv not found: {path}")
    return path


def find_edges_file(inst_dir: str, pct: int) -> str:
    path = os.path.join(inst_dir, f"solver_edges_top_{pct}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"solver_edges_top_{pct}.csv not found: {path}")
    return path


# ============================================================
# Loading data
# ============================================================
def load_nodes(nodes_path: str):
    df = pd.read_csv(nodes_path)

    node_id_col = find_col(df, ["node_id", "id", "node"])
    demand_col = find_col(df, ["demand"])
    ready_col = find_col(df, ["ready_time", "tw_start", "earliest"])
    due_col = find_col(df, ["due_time", "tw_end", "latest"])
    service_col = find_col(df, ["service_time", "service", "service_duration"])
    is_depot_col = find_col(df, ["is_depot", "depot"], required=False)
    x_col = find_col(df, ["x", "x_coord", "coord_x"], required=False)
    y_col = find_col(df, ["y", "y_coord", "coord_y"], required=False)

    nodes = {}
    depot = None

    for _, row in df.iterrows():
        nid = safe_int(row[node_id_col])
        demand = safe_float(row[demand_col], 0.0)
        ready = safe_float(row[ready_col], 0.0)
        due = safe_float(row[due_col], 0.0)
        service = safe_float(row[service_col], 0.0)

        if is_depot_col is not None:
            is_depot = safe_int(row[is_depot_col], 0)
        else:
            is_depot = 1 if nid == 0 else 0

        x = safe_float(row[x_col], 0.0) if x_col is not None else None
        y = safe_float(row[y_col], 0.0) if y_col is not None else None

        nodes[nid] = {
            "id": nid,
            "demand": demand,
            "ready": ready,
            "due": due,
            "service": service,
            "is_depot": is_depot,
            "x": x,
            "y": y,
        }

        if is_depot == 1:
            depot = nid

    if depot is None:
        depot = 0 if 0 in nodes else min(nodes.keys())

    node_ids = sorted(nodes.keys())
    customers = [n for n in node_ids if n != depot]

    return {
        "nodes": nodes,
        "node_ids": node_ids,
        "customers": customers,
        "depot": depot,
    }


def euclidean(a: dict, b: dict) -> float:
    if a["x"] is None or a["y"] is None or b["x"] is None or b["y"] is None:
        raise ValueError("Cannot compute distance: x/y missing in nodes.csv")
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def load_edges(edges_path: str, nodes: Dict[int, dict]):
    df = pd.read_csv(edges_path)

    from_col = find_col(df, ["from", "src", "i", "source"])
    to_col = find_col(df, ["to", "dst", "j", "target"])

    cost_col = find_col(df, ["cost", "distance", "dist"], required=False)
    travel_col = find_col(df, ["travel_time", "time", "tt"], required=False)

    arcs = {}

    for _, row in df.iterrows():
        i = safe_int(row[from_col])
        j = safe_int(row[to_col])

        if i == j:
            continue
        if i not in nodes or j not in nodes:
            continue

        if cost_col is not None:
            cost = safe_float(row[cost_col], 0.0)
        else:
            cost = euclidean(nodes[i], nodes[j])

        if travel_col is not None:
            travel = safe_float(row[travel_col], cost)
        else:
            travel = cost

        arcs[(i, j)] = {
            "cost": cost,
            "travel": travel,
        }

    return arcs


def ensure_depot_edges(node_data: dict, arcs: Dict[tuple, dict]):
    depot = node_data["depot"]
    customers = node_data["customers"]
    nodes = node_data["nodes"]

    for c in customers:
        if (depot, c) not in arcs:
            dist = euclidean(nodes[depot], nodes[c])
            arcs[(depot, c)] = {"cost": dist, "travel": dist}

        if (c, depot) not in arcs:
            dist = euclidean(nodes[c], nodes[depot])
            arcs[(c, depot)] = {"cost": dist, "travel": dist}

    return arcs


def load_instance(inst_dir: str, pct: int, capacity: float):
    nodes_path = find_nodes_file(inst_dir)
    edges_path = find_edges_file(inst_dir, pct)

    node_data = load_nodes(nodes_path)
    arcs = load_edges(edges_path, node_data["nodes"])
    arcs = ensure_depot_edges(node_data, arcs)

    return {
        "inst_dir": inst_dir,
        "pct": pct,
        "capacity": capacity,
        "nodes": node_data["nodes"],
        "node_ids": node_data["node_ids"],
        "customers": node_data["customers"],
        "depot": node_data["depot"],
        "arcs": arcs,
        "nodes_path": nodes_path,
        "edges_path": edges_path,
    }


# ============================================================
# Precheck
# ============================================================
def quick_precheck(data: dict, max_vehicles: int):
    depot = data["depot"]
    customers = data["customers"]
    arcs = data["arcs"]
    nodes = data["nodes"]

    incoming = {c: 0 for c in customers}
    outgoing = {c: 0 for c in customers}
    dep_out = 0
    dep_in = 0

    for (i, j) in arcs.keys():
        if j in incoming:
            incoming[j] += 1
        if i in outgoing:
            outgoing[i] += 1
        if i == depot and j != depot:
            dep_out += 1
        if i != depot and j == depot:
            dep_in += 1

    no_in = [c for c in customers if incoming[c] == 0]
    no_out = [c for c in customers if outgoing[c] == 0]

    if no_in:
        return False, f"Customers with no incoming edges: {no_in[:10]}"
    if no_out:
        return False, f"Customers with no outgoing edges: {no_out[:10]}"
    if dep_out == 0:
        return False, "No depot->customer edges"
    if dep_in == 0:
        return False, "No customer->depot edges"

    total_demand = sum(nodes[c]["demand"] for c in customers)
    if total_demand > max_vehicles * data["capacity"]:
        return False, f"Total demand {total_demand} exceeds max fleet capacity {max_vehicles * data['capacity']}"

    return True, "OK"


# ============================================================
# Build model
# ============================================================
def build_model(data: dict, time_limit: int, mip_gap: float, max_vehicles: int, verbose: bool, output_flag: int = 0):
    depot = data["depot"]
    customers = data["customers"]
    node_ids = data["node_ids"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    Q = data["capacity"]

    A = list(arcs.keys())

    model = gp.Model(f"VRPTW_top_{data['pct']}")
    model.Params.OutputFlag = output_flag
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    x = model.addVars(A, vtype=GRB.BINARY, name="x")
    t = model.addVars(node_ids, vtype=GRB.CONTINUOUS, lb=0.0, name="t")
    u = model.addVars(node_ids, vtype=GRB.CONTINUOUS, lb=0.0, ub=Q, name="u")

    model.setObjective(gp.quicksum(arcs[a]["cost"] * x[a] for a in A), GRB.MINIMIZE)

    for c in customers:
        in_arcs = [a for a in A if a[1] == c]
        out_arcs = [a for a in A if a[0] == c]
        model.addConstr(gp.quicksum(x[a] for a in in_arcs) == 1, name=f"in_{c}")
        model.addConstr(gp.quicksum(x[a] for a in out_arcs) == 1, name=f"out_{c}")

    dep_out = [a for a in A if a[0] == depot and a[1] != depot]
    dep_in = [a for a in A if a[1] == depot and a[0] != depot]

    model.addConstr(gp.quicksum(x[a] for a in dep_out) <= max_vehicles, name="max_vehicle_out")
    model.addConstr(gp.quicksum(x[a] for a in dep_in) <= max_vehicles, name="max_vehicle_in")
    model.addConstr(gp.quicksum(x[a] for a in dep_out) == gp.quicksum(x[a] for a in dep_in), name="vehicle_balance")

    for i in node_ids:
        model.addConstr(t[i] >= nodes[i]["ready"], name=f"ready_{i}")
        model.addConstr(t[i] <= nodes[i]["due"], name=f"due_{i}")

    model.addConstr(t[depot] >= nodes[depot]["ready"], name="depot_ready")

    for (i, j) in A:
        if j == depot:
            continue

        M = max(1e5, nodes[i]["due"] + nodes[i]["service"] + arcs[(i, j)]["travel"] - nodes[j]["ready"])
        model.addConstr(
            t[j] >= t[i] + nodes[i]["service"] + arcs[(i, j)]["travel"] - M * (1 - x[(i, j)]),
            name=f"time_{i}_{j}"
        )

    model.addConstr(u[depot] == 0.0, name="load_depot")

    for c in customers:
        model.addConstr(u[c] >= nodes[c]["demand"], name=f"load_lb_{c}")
        model.addConstr(u[c] <= Q, name=f"load_ub_{c}")

    for (i, j) in A:
        if i == depot and j != depot:
            model.addConstr(u[j] >= nodes[j]["demand"] * x[(i, j)], name=f"cap_from_depot_{j}")
        elif i != depot and j != depot:
            model.addConstr(u[j] >= u[i] + nodes[j]["demand"] - Q * (1 - x[(i, j)]), name=f"cap_{i}_{j}")

    return model, x


# ============================================================
# Extract routes
# ============================================================
def extract_solution(model, x, data):
    if model.SolCount == 0:
        return None

    depot = data["depot"]
    selected = [(i, j) for (i, j) in data["arcs"].keys() if x[(i, j)].X > 0.5]

    succ = {}
    for i, j in selected:
        succ.setdefault(i, []).append(j)

    routes = []
    if depot in succ:
        for first in succ[depot]:
            route = [depot, first]
            cur = first
            seen = {depot, first}

            while cur != depot:
                nxts = succ.get(cur, [])
                if not nxts:
                    break
                nxt = nxts[0]
                route.append(nxt)
                if nxt == depot:
                    break
                if nxt in seen:
                    break
                seen.add(nxt)
                cur = nxt

            routes.append(route)

    return {
        "objective": float(model.ObjVal),
        "num_routes": len(routes),
        "selected_edges": selected,
        "routes": routes,
    }


# ============================================================
# Dynamic solver
# ============================================================
def solve_instance_dynamic(
    inst_dir: str,
    subsets_to_try: List[int],
    capacity: float,
    time_limit: int,
    mip_gap: float,
    max_vehicles: int,
    verbose: bool = False,
    stop_on_feasible: bool = True,
    update_fn=None,
    output_flag: int = 0,
):
    history = []

    for pct in subsets_to_try:
        t0 = time.time()
        
        if update_fn:
            update_fn(f"Testing [bold cyan]{pct}%[/bold cyan] edge subset...")

        try:
            data = load_instance(inst_dir=inst_dir, pct=pct, capacity=capacity)
            ok, msg = quick_precheck(data, max_vehicles=max_vehicles)

            if not ok:
                if update_fn:
                    update_fn(f"[yellow]Skipped {pct}% (Precheck)[/yellow]")
                history.append({
                    "pct": pct,
                    "status": "SKIPPED_PRECHECK",
                    "message": msg,
                    "runtime_sec": round(time.time() - t0, 4),
                })
                continue

            model, x = build_model(
                data=data,
                time_limit=time_limit,
                mip_gap=mip_gap,
                max_vehicles=max_vehicles,
                verbose=verbose,
                output_flag=output_flag,
            )
            
            def gurobi_callback(model, where):
                if where == GRB.Callback.MIP:
                    objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                    objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                    
                    gap = 0.0
                    if abs(objbst) > 1e-10:
                        gap = abs(objbst - objbnd) / abs(objbst)
                    
                    if update_fn and objbst < 1e30:
                        update_fn(f"Optimizing [bold cyan]{pct}%[/bold cyan] (Gap: [bold yellow]{gap*100:.2f}%[/bold yellow], Obj: [bold green]{objbst:.2f}[/bold green])")

            if update_fn:
                update_fn(f"Optimizing [bold cyan]{pct}%[/bold cyan]...")

            model.optimize(gurobi_callback)

            status_map = {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.SUBOPTIMAL: "SUBOPTIMAL",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
                GRB.UNBOUNDED: "UNBOUNDED",
                GRB.INTERRUPTED: "INTERRUPTED",
            }
            status_name = status_map.get(model.Status, str(model.Status))

            rec = {
                "pct": pct,
                "status": status_name,
                "runtime_sec": round(time.time() - t0, 4),
                "sol_count": int(model.SolCount),
                "nodes_file": data["nodes_path"],
                "edges_file": data["edges_path"],
                "num_nodes": len(data["node_ids"]),
                "num_edges": len(data["arcs"]),
                "gap": model.MIPGap if model.SolCount > 0 else None,
            }

            if model.SolCount > 0:
                rec["objective"] = float(model.ObjVal)
                sol = extract_solution(model, x, data)
                rec["solution"] = sol
                rec["num_routes"] = sol["num_routes"] if sol else 0

            history.append(rec)

            if stop_on_feasible and model.SolCount > 0:
                return {
                    "success": True,
                    "chosen_pct": pct,
                    "history": history,
                    "final": rec,
                }

        except Exception as e:
            history.append({
                "pct": pct,
                "status": "ERROR",
                "error": str(e),
                "runtime_sec": round(time.time() - t0, 4),
            })

    if verbose and history:
        table = Table(title=f"Subset attempts for {os.path.basename(inst_dir)}", box=box.SIMPLE_HEAD)
        table.add_column("Pct", justify="right")
        table.add_column("Status")
        table.add_column("Time (s)", justify="right")
        table.add_column("Sols", justify="right")
        table.add_column("Vehicles", justify="right", style="yellow")
        table.add_column("Objective", justify="right")

        for h in history:
            color = "green" if h.get("sol_count", 0) > 0 else "red"
            status = h.get("status", "UNKNOWN")
            if status == "SKIPPED_PRECHECK":
                color = "yellow"
            
            table.add_row(
                str(h["pct"]),
                f"[{color}]{status}[/{color}]",
                f"{h['runtime_sec']:.2f}",
                str(h.get("sol_count", 0)),
                str(h.get("num_routes", "-")),
                f"{h.get('objective', 0.0):.2f}" if "objective" in h else "-"
            )
        console.print(table)

    last = history[-1] if history else None
    success = any(h.get("sol_count", 0) > 0 for h in history)

    return {
        "success": success,
        "chosen_pct": next((h["pct"] for h in history if h.get("sol_count", 0) > 0), None),
        "history": history,
        "final": last,
    }


# ============================================================
# Main
# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Simple dynamic Gurobi solver")
    parser.add_argument("--solver_graph_root", type=str, default=os.path.join(script_dir, "solver_graph"))
    parser.add_argument("--output_root", type=str, default=os.path.join(script_dir, "solver_output"))
    parser.add_argument("--capacity", type=float, default=200.0)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--mip_gap", type=float, default=0.01)
    parser.add_argument("--max_vehicles", type=int, default=30)
    parser.add_argument("--subsets", type=int, nargs="+", default=[15, 20, 25, 30, 40, 50])
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--stop_on_feasible", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gurobi_verbose", action="store_true", help="Enable verbose Gurobi output")
    args = parser.parse_args()

    output_flag = 1 if args.gurobi_verbose else 0
    ensure_dir(args.output_root)

    all_instances = discover_instances(args.solver_graph_root, args.instances)
    if not all_instances:
        raise FileNotFoundError(f"No instances found under: {args.solver_graph_root}")

    summary_rows = []

    # Prepare real-time display
    job_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    total_task = job_progress.add_task("[cyan]Overall Progress", total=len(all_instances))

    status_table = Table(title="Live Solver Status", box=box.SIMPLE)
    status_table.add_column("Instance", style="cyan")
    status_table.add_column("Current Task", style="yellow")
    status_table.add_column("Status", justify="center")

    instance_statuses = {} # Name -> Current Task/Status text

    def update_live_table():
        new_table = Table(title="Live Solver Status", box=box.SIMPLE)
        new_table.add_column("Instance", style="cyan")
        new_table.add_column("Current Task", style="yellow")
        new_table.add_column("Status", justify="center")
        
        # Show top 5 recent/active instances to keep it clean
        displayed = list(instance_statuses.keys())[-10:]
        for inst in displayed:
            task_text, status_icon = instance_statuses[inst]
            new_table.add_row(inst, task_text, status_icon)
        return new_table

    with Live(Panel(job_progress, title="[bold]VRPTW Solver Dashboard[/bold]", border_style="blue"), console=console, refresh_per_second=4) as live:
        
        for inst_name, inst_dir in all_instances:
            instance_statuses[inst_name] = ("Waiting...", "⏳")
            
            # Sub-task update function
            def make_update_fn(name):
                def up(msg):
                    instance_statuses[name] = (msg, "🚀")
                    live.update(
                        Panel(
                            Group(job_progress, update_live_table()),
                            title="[bold]VRPTW Solver Dashboard[/bold]",
                            border_style="blue"
                        )
                    )
                return up

            instance_statuses[inst_name] = ("Starting...", "🚀")
            live.update(Panel(Group(job_progress, update_live_table()), title="[bold]VRPTW Solver Dashboard[/bold]", border_style="blue"))
            
            # Clear demarcator for instance start
            live.console.print("\n")
            live.console.print(Rule(f"[bold yellow]STARTING INSTANCE: {inst_name}[/bold yellow]", style="yellow", align="left"))
            
            # Pre-load metadata for the header
            try:
                temp_nodes_path = find_nodes_file(inst_dir)
                temp_node_data = load_nodes(temp_nodes_path)
                live.console.print(f"[dim]Directory: {inst_dir}[/dim]")
                live.console.print(f"[bold cyan]Nodes:[/bold cyan] {len(temp_node_data['node_ids'])} | [bold cyan]Depot:[/bold cyan] {temp_node_data['depot']} | [bold cyan]Capacity:[/bold cyan] {args.capacity}\n")
            except Exception:
                live.console.print(f"[dim]Directory: {inst_dir}[/dim]\n")

            result = solve_instance_dynamic(
                inst_dir=inst_dir,
                subsets_to_try=args.subsets,
                capacity=args.capacity,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                max_vehicles=args.max_vehicles,
                verbose=args.verbose,
                stop_on_feasible=args.stop_on_feasible,
                update_fn=make_update_fn(inst_name),
                output_flag=output_flag
            )

            success = result.get("success", False)
            instance_statuses[inst_name] = ("Finished", "✅" if success else "❌")
            
            job_progress.advance(total_task)
            live.update(Panel(Group(job_progress, update_live_table()), title="[bold]VRPTW Solver Dashboard[/bold]", border_style="blue"))

            out_dir = os.path.join(args.output_root, inst_name)
            ensure_dir(out_dir)
            write_json(os.path.join(out_dir, "dynamic_result.json"), result)

            final = result.get("final", {}) or {}
            summary_rows.append({
                "instance": inst_name,
                "success": success,
                "chosen_pct": result.get("chosen_pct"),
                "final_status": final.get("status"),
                "objective": final.get("objective"),
                "runtime_sec": final.get("runtime_sec"),
                "sol_count": final.get("sol_count"),
                "num_routes": final.get("num_routes"),
            })

            # Print per-instance report to the scrolling live log
            report_table = Table(title=f"Completed: {inst_name}", box=box.ROUNDED, title_justify="left", border_style="green" if success else "red")
            report_table.add_column("Field", style="bold cyan")
            report_table.add_column("Details", style="white")
            
            report_table.add_row("Final Status", f"[bold]{final.get('status', 'ERROR')}[/bold]")
            report_table.add_row("Edges Used", f"{result.get('chosen_pct', 'None')}%")
            report_table.add_row("Objective", f"{final.get('objective', 0.0):.2f}" if final.get("objective") is not None else "N/A")
            report_table.add_row("Gap Reached", f"{final.get('gap', 0.0)*100:.2f}%" if final.get('gap') is not None else "N/A")
            report_table.add_row("Vehicles", str(final.get("num_routes", "N/A")))
            report_table.add_row("Nodes / Edges", f"{final.get('num_nodes', 'N/A')} / {final.get('num_edges', 'N/A')}")
            report_table.add_row("Solve Time", f"{final.get('runtime_sec', 0.0):.2f}s")
            
            live.console.print(Rule(f"[bold green]COMPLETED INSTANCE: {inst_name}[/bold green]", style="green", align="left"))
            live.console.print(report_table)
            live.console.print("\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    console.print("\n[bold green]Batch Processing Completed.[/bold green]")
    console.print(f"Full summary saved to: [cyan]{summary_csv}[/cyan]")

    # Final summary table
    table = Table(title="Optimization Summary", box=box.ROUNDED)
    table.add_column("Instance", style="cyan")
    table.add_column("Success", justify="center")
    table.add_column("Pct (%)", justify="right")
    table.add_column("Status", style="magenta")
    table.add_column("Objective", justify="right", style="green")
    table.add_column("Vehicles", justify="right", style="yellow")
    table.add_column("Time (s)", justify="right")
    table.add_column("Sols", justify="right")

    for row in summary_rows:
        success_str = "[green]Yes[/green]" if row["success"] else "[red]No[/red]"
        table.add_row(
            row["instance"],
            success_str,
            str(row["chosen_pct"] or "-"),
            str(row["final_status"] or "-"),
            f"{row['objective']:.2f}" if row["objective"] is not None else "-",
            str(row["num_routes"] or "-"),
            f"{row['runtime_sec']:.2f}" if row["runtime_sec"] is not None else "-",
            str(row["sol_count"] or 0)
        )
    console.print(table)


if __name__ == "__main__":
    main()