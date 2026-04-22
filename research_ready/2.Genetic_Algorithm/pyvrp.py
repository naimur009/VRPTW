import argparse
import csv
import math
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# WORKAROUND: Allow script to be named pyvrp.py without circular import
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

# Rich for beautiful terminal dashboard
try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    console = Console()
except Exception:
    console = None
    # Fallback to no-rich mode

# ──────────────────────────────────────────────
# Dashboard State Manager
# ──────────────────────────────────────────────
class Dashboard:
    def __init__(self, total_files):
        self.layout = Layout()
        self.layout.split_row(
            Layout(name="left", ratio=10),
            Layout(name="right", ratio=12)
        )
        
        # Left side structure
        self.layout["left"].split_column(
            Layout(name="info", size=9),
            Layout(name="progression", size=10),
            Layout(name="logs")
        )
        
        # Right side structure
        self.layout["right"].split_column(
            Layout(name="stats", size=8),
            Layout(name="results")
        )

        # Right side: History Table
        self.results_table = Table(
            title="Processing History",
            expand=True,
            header_style="bold magenta"
        )
        self.results_table.add_column("Instance", style="cyan")
        self.results_table.add_column("Dist", justify="right", style="green")
        self.results_table.add_column("Veh", justify="center", style="blue")
        self.results_table.add_column("Veh T", justify="right", style="yellow")
        self.results_table.add_column("Dist T", justify="right", style="yellow")
        self.results_table.add_column("Total", justify="right", style="bold yellow")

        # CSV Logger Setup
        base_dir = Path(__file__).parent.absolute() / "dataset"
        base_dir.mkdir(exist_ok=True)
        self.csv_path = base_dir / "batch_summary.csv"
        self.init_csv()

        # Right side: Overall Progress
        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        self.total_files = total_files
        self.completed_count = 0
        self.error_count = 0
        self.overall_task = self.overall_progress.add_task("[bold green]Batch Progress", total=total_files)

        # Left side: Current Progress with Timers
        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        self.inst_task = None

        # Metadata State
        self.base_metadata = []
        self.veh_time = None
        self.dist_time = None

        self.log_messages = []
        self.current_inst_name = "Waiting..."
        self.start_time = time.time()
        
        self.update_layout()

    def init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Instance", "Distance", "Vehicles", "Veh Time (s)", "Dist Time (s)", "Total Time (s)", "Timestamp"])

    def reset_current(self):
        self.log_messages = []
        self.veh_time = None
        self.dist_time = None
        self.base_metadata = []
        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

    def add_log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {msg}")
        if len(self.log_messages) > 8:
            self.log_messages.pop(0)
        self.update_layout()

    def update_current(self, inst_name, data, ph1_limit, ph2_limit):
        self.reset_current()
        self.current_inst_name = inst_name
        self.inst_task = self.current_progress.add_task(f"[bold cyan]Current: {inst_name}", total=100)
        
        # Store base metadata
        self.base_metadata = [
            ("Customers:", str(data.num_clients)),
            ("Locations:", f"{data.num_locations} (incl. depot)"),
            ("Phase 1 Limit:", f"{ph1_limit}s"),
            ("Phase 2 Limit:", f"{ph2_limit}s"),
            ("Vehicles Available:", str(data.vehicle_type(0).num_available)),
            ("Capacity:", str(data.vehicle_type(0).capacity))
        ]

        self.add_log(f"Started [bold yellow]{inst_name}[/bold yellow]")
        self.update_layout()

    def start_phase_task(self, description, total=100):
        return self.current_progress.add_task(f"  [white]{description}", total=total)

    def finish_phase_task(self, task_id, status="OK"):
        desc = self.current_progress.tasks[task_id].description
        new_desc = desc.replace("[white]", "[bold green]✓ [/bold green][white]") + f" [bold green]({status})[/bold green]"
        self.current_progress.update(task_id, description=new_desc, completed=100)
        self.current_progress.stop_task(task_id)
        self.update_layout()

    def set_phase_time(self, veh_time=None, dist_time=None):
        if veh_time is not None: self.veh_time = veh_time
        if dist_time is not None: self.dist_time = dist_time
        self.update_layout()

    def add_result(self, name, dist, vehicles, veh_time, dist_time, total_time):
        # Update Table
        self.results_table.add_row(
            name,
            f"{dist:.2f}",
            str(vehicles),
            f"{veh_time:.1f}s",
            f"{dist_time:.1f}s",
            f"{total_time:.1f}s"
        )
        
        # Persistent CSV Logging
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                name, 
                f"{dist:.4f}", 
                vehicles, 
                f"{veh_time:.2f}", 
                f"{dist_time:.2f}", 
                f"{total_time:.2f}", 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        self.completed_count += 1
        self.overall_progress.advance(self.overall_task)
        if self.inst_task is not None:
             self.current_progress.update(self.inst_task, completed=100)
        self.update_layout()

    def add_error(self, name, error_msg):
        self.error_count += 1
        self.overall_progress.advance(self.overall_task)
        self.add_log(f"[bold red]Error {name}:[/bold red] {error_msg}")
        
        # Log error to CSV as well
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
        self.update_layout()

    def update_layout(self):
        # Left Panel rebuilds
        metadata_table = Table.grid(padding=(0, 1))
        metadata_table.add_column("Key", style="bold blue")
        metadata_table.add_column("Value", style="white")
        for k, v in self.base_metadata:
            metadata_table.add_row(k, v)
        if self.veh_time is not None:
            metadata_table.add_row("Vehicle Find Time:", f"[bold yellow]{self.veh_time:.1f}s[/bold yellow]")
        if self.dist_time is not None:
            metadata_table.add_row("Dist Opt Time:", f"[bold yellow]{self.dist_time:.1f}s[/bold yellow]")

        # Right Panel rebuilds (Stats)
        pending = self.total_files - self.completed_count - self.error_count
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column("Stat", style="white")
        stats_table.add_column("Value", justify="right")
        stats_table.add_row("[bold green]Completed:[/bold green]", f"[bold green]{self.completed_count}[/bold green]")
        stats_table.add_row("[bold yellow]Pending:[/bold yellow]", f"[bold yellow]{pending}[/bold yellow]")
        stats_table.add_row("[bold red]Errors:[/bold red]", f"[bold red]{self.error_count}[/bold red]")
        stats_table.add_row("[bold blue]Total Files:[/bold blue]", f"[bold blue]{self.total_files}[/bold blue]")

        # Build Panels
        self.layout["info"].update(
            Panel(metadata_table, title=f"[bold blue]Instance Info: {self.current_inst_name}[/bold blue]", border_style="blue")
        )
        self.layout["progression"].update(
            Panel(self.current_progress, title=f"[bold cyan]Progression Checklist[/bold cyan]", border_style="cyan")
        )
        log_text = Text.from_markup("\n".join(self.log_messages))
        self.layout["logs"].update(
            Panel(log_text, title="[bold white]Event Log[/bold white]", border_style="white")
        )
        
        self.layout["stats"].update(
            Panel(
                Group(self.overall_progress, stats_table),
                title="[bold green]Overall Status[/bold green]",
                border_style="green"
            )
        )
        self.layout["results"].update(
            Panel(self.results_table, border_style="magenta")
        )


# ──────────────────────────────────────────────
# Helper: rebuild ProblemData with a capped fleet
# ──────────────────────────────────────────────
def _build_model_with_fleet(data, num_vehicles: int) -> Model:
    m = Model()
    new_locs = []
    depot = data.depots()[0]
    m_depot = m.add_depot(x=depot.x, y=depot.y, tw_early=depot.tw_early, tw_late=depot.tw_late)
    new_locs.append(m_depot)
    for client in data.clients():
        m_client = m.add_client(
            x=client.x, y=client.y, delivery=client.delivery,
            service_duration=client.service_duration,
            tw_early=client.tw_early, tw_late=client.tw_late
        )
        new_locs.append(m_client)
    dist_mat = data.distance_matrix(0)
    dur_mat  = data.duration_matrix(0)
    for i in range(data.num_locations):
        for j in range(data.num_locations):
            if i == j: continue
            m.add_edge(new_locs[i], new_locs[j], distance=int(dist_mat[i, j]), duration=int(dur_mat[i, j]))
    vt = data.vehicle_type(0)
    m.add_vehicle_type(num_available=num_vehicles, capacity=vt.capacity)
    return m


# ──────────────────────────────────────────────
# STEP 1 – Minimise the number of vehicles
# ──────────────────────────────────────────────
def step1_minimize_vehicles(data, time_limit: float, dashboard=None):
    task_id = None
    if dashboard:
        dashboard.add_log("Step 1: Minimizing vehicles...")
        task_id = dashboard.start_phase_task("Step 1: Vehicle Minimization")
    
    result = solve(data, stop=MaxRuntime(time_limit), display=False)
    if not result.best.is_feasible():
        raise RuntimeError("No feasible solution found.")
    
    best_sol = result.best
    min_k    = best_sol.num_routes()
    start_k  = min_k
    
    if dashboard: dashboard.add_log(f"Initial fleet: [bold blue]{min_k}[/bold blue]")

    while min_k > 1:
        candidate_k = min_k - 1
        if dashboard and task_id is not None:
            if start_k > 0:
                 pct = 100 * (start_k - candidate_k) / start_k
                 dashboard.current_progress.update(task_id, completed=pct)
        
        m = _build_model_with_fleet(data, candidate_k)
        new_result = solve(m.data(), stop=MaxRuntime(time_limit), display=False)
        new_sol = new_result.best

        if new_sol.is_feasible():
            best_sol = new_sol
            min_k    = candidate_k
        else:
            if dashboard: dashboard.add_log(f"✘ [red]{candidate_k}[/red] infeasible")
            break

    if dashboard and task_id is not None:
        dashboard.finish_phase_task(task_id, f"{min_k} Veh")
    return min_k, best_sol


# ──────────────────────────────────────────────
# STEP 2 – Minimise distance for the fixed min_k
# ──────────────────────────────────────────────
def step2_minimize_distance(data, min_k: int, time_limit: float, dashboard=None):
    task_id = None
    if dashboard:
        dashboard.add_log(f"Step 2: Optimizing (k={min_k})...")
        task_id = dashboard.start_phase_task(f"Step 2: Distance Optimization (k={min_k})")
    
    m = _build_model_with_fleet(data, min_k)
    result = solve(m.data(), stop=MaxRuntime(time_limit), display=False)
    
    if dashboard and task_id is not None:
        dashboard.finish_phase_task(task_id, "OK")
    return result.best


# ──────────────────────────────────────────────
# Instance Processing
# ──────────────────────────────────────────────
def _read_csv(path: Path) -> Model:
    m = Model()
    with open(path, "r") as f:
        all_lines = f.readlines()
    data_lines = [ln for ln in all_lines if not ln.startswith("#")]
    num_vehicles, capacity = 25, 200
    for ln in all_lines:
        if ln.startswith("#"):
            import re
            v = re.search(r"n_vehicles=(\d+)", ln.lower())
            c = re.search(r"capacity=(\d+)", ln.lower())
            if v: num_vehicles = int(v.group(1))
            if c: capacity = int(c.group(1))

    import io
    reader = csv.DictReader(io.StringIO("".join(data_lines)))
    rows = list(reader)
    def get_val(r, ks):
        for k in r.keys():
            if k.strip().upper() in [s.upper() for s in ks]: return r[k]
        return 0
    depot = rows[0]
    m.add_depot(
        x=round(float(get_val(depot, ["XCOORD.", "x"])) * 10),
        y=round(float(get_val(depot, ["YCOORD.", "y"])) * 10),
        tw_early=round(float(get_val(depot, ["READY TIME", "ready", "tw_start"])) * 10),
        tw_late=round(float(get_val(depot, ["DUE DATE", "due", "tw_end"])) * 10)
    )
    for r in rows[1:]:
        m.add_client(
            x=round(float(get_val(r, ["XCOORD.", "x"])) * 10),
            y=round(float(get_val(r, ["YCOORD.", "y"])) * 10),
            delivery=[int(get_val(r, ["DEMAND", "demand", "q"]))],
            service_duration=round(float(get_val(r, ["SERVICE TIME", "service"])) * 10),
            tw_early=round(float(get_val(r, ["READY TIME", "ready", "tw_start"])) * 10),
            tw_late=round(float(get_val(r, ["DUE DATE", "due", "tw_end"])) * 10)
        )
    lcs = m.locations
    for i in lcs:
        for j in lcs:
            if i is j: continue
            d = round(math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2))
            m.add_edge(i, j, distance=d, duration=d)
    m.add_vehicle_type(num_available=num_vehicles, capacity=capacity)
    return m

def _parse_folder_name(stem: str) -> str:
    """
    Convert a CSV stem like  R1_n100_d75_0001
    into a folder name like  inst_R1_D75_0001

    Expected format: {SERIES}_n{N}_d{DENSITY}_{IDX}
    Falls back to  inst_{stem}  if the pattern doesn't match.
    """
    import re
    m = re.fullmatch(
        r'([A-Za-z0-9]+)_n(\d+)_d(\d+)_(\d+)',
        stem
    )
    if m:
        series  = m.group(1).upper()   # e.g. R1
        density = m.group(3)            # e.g. 75
        idx     = m.group(4)            # e.g. 0001
        return f"inst_{series}_D{density}_{idx}"
    return f"inst_{stem}"


def save_output(instance_path: str, sol):
    base_dir = Path(__file__).parent.absolute() / "dataset"
    base_dir.mkdir(exist_ok=True)
    source_path = Path(instance_path).absolute()

    # Mirror data/ folder structure: use the source file's parent folder name (e.g. "R1")
    series_folder = source_path.parent.name  # e.g. "R1", "C2", "RC1"

    # Build structured instance folder name from the CSV filename
    folder_name = _parse_folder_name(source_path.stem)  # e.g. "inst_R1_D25_0001"
    folder_path = base_dir / series_folder / folder_name

    # If the folder already exists, add a numeric suffix to avoid collisions
    suffix = 1
    base_folder_name = folder_name
    while folder_path.exists():
        folder_path = base_dir / series_folder / f"{base_folder_name}_{suffix}"
        suffix += 1

    folder_path.mkdir(parents=True)

    # Copy the source CSV into the instance folder
    import shutil
    dest_name = "data.csv" if source_path.suffix.lower() == ".csv" else "main.txt"
    if source_path.exists():
        shutil.copy2(str(source_path), str(folder_path / dest_name))

    sol_lines = [f"Route {i+1} : {' '.join(str(v) for v in r)}" for i, r in enumerate(sol.routes())]
    with open(folder_path / "solution.txt", "w") as f:
        f.write("\n".join(sol_lines))
    # Return a human-readable relative path for logging
    return f"{series_folder}/{folder_name}"

def solve_vrptw(filepath: str, ph1_time: float, ph2_time: float, dashboard=None):
    start = time.time()
    path = Path(filepath)
    data = _read_csv(path).data() if path.suffix.lower() == ".csv" else read(filepath)
    if dashboard: dashboard.update_current(path.name, data, ph1_time, ph2_time)
    
    v_start = time.time()
    min_k, _ = step1_minimize_vehicles(data, ph1_time, dashboard)
    veh_time = time.time() - v_start
    if dashboard: dashboard.set_phase_time(veh_time=veh_time)
    
    d_start = time.time()
    final_sol = step2_minimize_distance(data, min_k, ph2_time, dashboard)
    dist_time = time.time() - d_start
    if dashboard: dashboard.set_phase_time(dist_time=dist_time)
    
    folder_name = save_output(filepath, final_sol)
    total_time = time.time() - start
    
    if dashboard:
        dashboard.add_result(path.name, final_sol.distance()/10, final_sol.num_routes(), veh_time, dist_time, total_time)
        dashboard.add_log(f"Saved in [bold blue]{folder_name}[/bold blue]")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VRPTW Dashboard Solver")
    parser.add_argument("instance", nargs="?", default=None)
    parser.add_argument("--phase1-time", type=float, default=10.0)
    parser.add_argument("--phase2-time", type=float, default=30.0)
    args = parser.parse_args()

    if not args.instance:
        # Default: recursively scan the 'data/' folder next to this script
        data_root = Path(__file__).parent.absolute() / "data"
        if data_root.is_dir():
            files = sorted(data_root.rglob("*.csv"))
        else:
            # Fallback: CSVs in the current directory
            files = sorted(Path(".").glob("*.csv"))
        # Exclude anything already inside dataset output folders
        files = [f for f in files if "inst_" not in f.parent.name]
    else:
        p = Path(args.instance)
        if p.is_dir():
            files = sorted(p.rglob("*.csv"))
        else:
            files = [p]

    if not files:
        print("No instances found.")
        sys.exit(0)

    if console:
        db = Dashboard(len(files))
        with Live(db.layout, refresh_per_second=4, screen=True):
            for fpath in files:
                try:
                    solve_vrptw(str(fpath), args.phase1_time, args.phase2_time, db)
                except Exception as e:
                    db.add_error(fpath.name, str(e))
        console.print(db.results_table)
    else:
        for fpath in files:
            solve_vrptw(str(fpath), args.phase1_time, args.phase2_time)