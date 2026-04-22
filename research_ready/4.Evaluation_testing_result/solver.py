import os
import json
import math
import time
import argparse
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console, Group
from rich.rule import Rule
from rich.layout import Layout
from rich.columns import Columns

console = Console()


# ============================================================
# HUD State Manager
# ============================================================
class HUDState:
    def __init__(self, total_files):
        self.total_files = total_files
        self.completed = 0
        self.pending = total_files
        self.errors = 0
        
        # Current Instance
        self.inst_name = "Waiting..."
        self.nodes = 0
        self.edges = 0
        self.capacity = 0
        self.max_veh = 0
        self.t_limit = 0
        
        # Timers
        self.batch_start_time = time.time()
        self.inst_start_time = time.time()
        
        # Current Progress
        self.phase_name = "Idle"
        self.phase_pct = 0.0
        
        # History
        self.history = []
        
        # Event Log
        self.event_log = deque(maxlen=10)
        self.event_log.append("Solver Dashboard Initialized")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] {message}")

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=9),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].split_row(
            Layout(name="constraints", ratio=1),
            Layout(name="overall", ratio=1),
        )
        layout["bottom"].split_row(
            Layout(name="event_log", ratio=1),
            Layout(name="history", ratio=2),  # Give more room to the table
        )
        return layout

    def get_constraints_panel(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="white", justify="right")
        
        grid.add_row("Capacity:", f"{self.capacity}")
        grid.add_row("Max Vehicles:", f"{self.max_veh}")
        grid.add_row("Time Limit:", f"{self.t_limit}s")
        grid.add_row(Rule(style="dim"))
        grid.add_row("Cur Instance:", f"[bold yellow]{self.inst_name}[/bold yellow]")
        grid.add_row("Nodes:", str(self.nodes))
        grid.add_row("Edges:", str(self.edges))
        
        return Panel(grid, title="[bold blue]Solver Constraints & Instance[/bold blue]", border_style="blue")

    def get_overall_status_panel(self) -> Panel:
        batch_elapsed = time.time() - self.batch_start_time
        overall_pct = (self.completed / self.total_files) * 100 if self.total_files > 0 else 0
        
        grid = Table.grid(expand=True)
        grid.add_column(style="green")
        grid.add_column(style="white", justify="right")
        
        grid.add_row("Overall Progress:", f"{self.completed} / {self.total_files} ({overall_pct:.1f}%)")
        grid.add_row("Errors:", f"[red]{self.errors}[/red]")
        grid.add_row("Total Timer:", f"[bold white]{batch_elapsed:.1f}s[/bold white]")
        
        progress_bar = Progress(BarColumn(bar_width=None), TextColumn("[bold]{task.percentage:>3.0f}%"), console=console)
        p_task = progress_bar.add_task("overall", total=100, completed=overall_pct)
        
        content = Group(
            grid,
            Rule(style="dim"),
            progress_bar
        )
        return Panel(content, title="[bold green]Overall Progress[/bold green]", border_style="green")

    def get_event_log_panel(self) -> Panel:
        inst_elapsed = time.time() - self.inst_start_time
        
        progress_bar = Progress(BarColumn(bar_width=None), TextColumn("[bold]{task.percentage:>3.0f}%"), console=console)
        p_task = progress_bar.add_task("phase", total=100, completed=self.phase_pct)
        
        grid = Table.grid(expand=True)
        grid.add_row(f"[bold cyan]Status:[/bold cyan] {self.phase_name}")
        grid.add_row(f"[bold cyan]Timer:[/bold cyan]  [white]{inst_elapsed:.1f}s[/white]")
        
        log_content = "\n".join(self.event_log)
        
        content = Group(
            grid,
            Rule(style="dim"),
            progress_bar,
            Rule(title="Recent Events", style="dim"),
            Text(log_content, style="dim white")
        )
        return Panel(content, title="[bold cyan]Event Log & Current Info[/bold cyan]", border_style="cyan")

    def build_history_table(self, items):
        table = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold magenta")
        table.add_column("Instance", style="white")
        table.add_column("Dist", justify="right", style="green")
        table.add_column("Veh", justify="right", style="yellow")
        table.add_column("Gap", justify="right", style="blue")
        table.add_column("Time", justify="right", style="dim")
        
        for r in items:
            table.add_row(
                r["instance"],
                f"{r['objective']:.1f}" if r['objective'] else "-",
                str(r["routes"]) if r["routes"] else "-",
                f"{r['gap']*100:.1f}%" if r['gap'] else "-",
                f"{r['runtime']:.1f}s" if r['runtime'] else "-"
            )
        return table

    def get_history_panel(self) -> Panel:
        # Show last 15 rows in live view to fill the panel better
        table = self.build_history_table(self.history[-15:])
        return Panel(table, title="[bold magenta]Completed Instances[/bold magenta]", border_style="magenta")

    def render(self) -> Layout:
        l = self.make_layout()
        l["constraints"].update(self.get_constraints_panel())
        l["overall"].update(self.get_overall_status_panel())
        l["event_log"].update(self.get_event_log_panel())
        l["history"].update(self.get_history_panel())
        return l


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
    state=None,
):
    history = []

    for pct in subsets_to_try:
        t0 = time.time()
        
        if update_fn:
            update_fn(f"Sub {pct}%...", 0)

        try:
            data = load_instance(inst_dir=inst_dir, pct=pct, capacity=capacity)
            if state:
                state.nodes = len(data["node_ids"])
                state.edges = len(data["arcs"])

            ok, msg = quick_precheck(data, max_vehicles=max_vehicles)

            if not ok:
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
                    runtime = model.cbGet(GRB.Callback.RUNTIME)
                    pct_done = (runtime / time_limit) * 100 if time_limit > 0 else 0
                    
                    objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                    if update_fn and objbst < 1e30:
                        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                        gap = abs(objbst - objbnd) / max(abs(objbst), 1e-10)
                        update_fn(f"Sub {pct}% (Gap:{gap*100:.1f}%, Obj:{objbst:.1f})", pct_done)
                    elif update_fn:
                        update_fn(f"Sub {pct}% (Searching...)", pct_done)

            model.optimize(gurobi_callback)

            status_map = {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.TIME_LIMIT: "TIME_LIMIT",
            }
            status_name = status_map.get(model.Status, str(model.Status))

            rec = {
                "pct": pct,
                "status": status_name,
                "runtime_sec": round(time.time() - t0, 4),
                "sol_count": int(model.SolCount),
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
                return {"success": True, "chosen_pct": pct, "history": history, "final": rec}

            history.append(rec)

        except Exception as e:
            history.append({
                "pct": pct,
                "status": "ERROR",
                "error": str(e),
                "runtime_sec": round(time.time() - t0, 4),
            })

    return {
        "success": any(h.get("sol_count", 0) > 0 for h in history),
        "chosen_pct": next((h["pct"] for h in history if h.get("sol_count", 0) > 0), None),
        "history": history,
        "final": history[-1] if history else None,
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
    parser.add_argument("--time_limit", type=int, default=5)
    parser.add_argument("--mip_gap", type=float, default=0.01)
    parser.add_argument("--max_vehicles", type=int, default=30)
    parser.add_argument("--subsets", type=int, nargs="+", default=[15, 20, 25, 30, 40, 50])
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--stop_on_feasible", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gurobi_verbose", action="store_true")
    args = parser.parse_args()

    output_flag = 1 if args.gurobi_verbose else 0
    ensure_dir(args.output_root)

    all_instances = discover_instances(args.solver_graph_root, args.instances)
    if not all_instances:
        console.print(f"[bold red]Error:[/bold red] No instances found in {args.solver_graph_root}")
        return

    # ── Init HUD State ──────────────────────────────────────────────────────
    state = HUDState(total_files=len(all_instances))
    state.t_limit = args.time_limit
    state.capacity = args.capacity
    state.max_veh = args.max_vehicles

    # ── Live Execution ──────────────────────────────────────────────────────
    with Live(state.render(), console=console, refresh_per_second=4, screen=False) as live:
        for inst_name, inst_dir in all_instances:
            # Update Current Instance Info
            state.inst_name = inst_name
            state.inst_start_time = time.time()
            state.phase_name = "Loading..."
            state.phase_pct = 0
            
            try:
                temp_nodes_path = find_nodes_file(inst_dir)
                temp_node_data = load_nodes(temp_nodes_path)
                state.nodes = len(temp_node_data['node_ids'])
            except Exception:
                state.nodes = 0
            
            live.update(state.render())

            def update_fn(msg, pct):
                # Only log significant phase changes or "Searching..." starts
                if state.phase_name != msg and ("Sub" in msg and "Gap" not in msg):
                    state.log(f"{inst_name}: {msg}")
                state.phase_name = msg
                state.phase_pct = pct
                live.update(state.render())

            result = solve_instance_dynamic(
                inst_dir=inst_dir,
                subsets_to_try=args.subsets,
                capacity=args.capacity,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                max_vehicles=args.max_vehicles,
                verbose=args.verbose,
                stop_on_feasible=args.stop_on_feasible,
                update_fn=update_fn,
                output_flag=output_flag,
                state=state
            )

            success = result.get("success", False)
            if not success:
                state.errors += 1
                state.log(f"[red]Failed:[/red] {inst_name}")
            else:
                state.completed += 1
                final = result.get("final", {}) or {}
                obj = final.get("objective", 0)
                state.log(f"[green]Solved:[/green] {inst_name} (Obj: {obj:.1f})")

            state.pending -= 1
            
            # Save artifacts
            out_dir = os.path.join(args.output_root, inst_name)
            ensure_dir(out_dir)
            write_json(os.path.join(out_dir, "dynamic_result.json"), result)

            final = result.get("final", {}) or {}
            history_entry = {
                "instance": inst_name,
                "success": success,
                "chosen_pct": result.get("chosen_pct"),
                "status": final.get("status"),
                "objective": final.get("objective"),
                "runtime": final.get("runtime_sec"),
                "routes": final.get("num_routes"),
                "gap": final.get("gap"),
            }
            state.history.append(history_entry)
            
            live.update(state.render())

    # ── Final Report ────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(state.history)
    summary_csv = os.path.join(args.output_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    console.print(Rule(style="dim"))
    console.print(state.build_history_table(state.history))
    console.print(Panel(
        f"Batch Completed: [bold]{state.completed}/{state.total_files}[/bold] solved.\n"
        f"Full summary saved to: [cyan]{summary_csv}[/cyan]\n"
        f"Results saved to: [cyan]{args.output_root}[/cyan]",
        border_style="bright_blue",
        title="[bold blue]Final Summary[/bold blue]"
    ))


if __name__ == "__main__":
    main()