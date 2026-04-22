import os
import glob
import json
import math
import random
import argparse
import time
from typing import List, Tuple, Dict, Optional, Callable
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich.rule import Rule
    from rich import box
    from rich.text import Text
    from rich.layout import Layout
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# HUD State Manager (Refined Design)
# ============================================================
class TrainingHUD:
    def __init__(self, total_epochs: int, train_batches: int, val_batches: int, config: dict):
        self.total_epochs = total_epochs
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.config = config
        
        self.current_epoch = 0
        self.history = []
        self.best_score = 0.0
        self.lr = config.get("lr", 0.0)
        self.patience_left = config.get("patience_max", 0)
        
        self.train_progress = 0
        self.val_progress = 0
        self.status = "Initializing..."
        
        self.event_log = deque(maxlen=4)
        self.start_time = time.time()

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"[dim]{ts}[/dim] {message}")

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=10),
            Layout(name="middle", size=22),
            Layout(name="bottom", size=6),
        )
        layout["top"].split_row(
            Layout(name="params", size=35),
            Layout(name="current", ratio=1),
        )
        return layout

    def get_params_panel(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="white", justify="right")
        
        grid.add_row("Device:", str(self.config.get("device", "N/A")))
        grid.add_row("Model Dim:", f"{self.config.get('hidden_dim', 0)}")
        grid.add_row("Batch Size:", str(self.config.get("batch_size", 0)))
        grid.add_row("Edge Feats:", str(self.config.get("edge_feats", 0)))
        grid.add_row(Rule(style="dim"))
        grid.add_row("Train Graphs:", str(self.config.get("train_graphs", 0)))
        grid.add_row("Val Graphs:", str(self.config.get("val_graphs", 0)))
        grid.add_row("Loss Weights:", self.config.get("weights", "N/A"))
        grid.add_row(Rule(style="dim"))
        grid.add_row("Best Score:", f"[bold green]{self.best_score:.4f}[/bold green]")
        grid.add_row("Cur LR:", f"{self.lr:.2e}")
        
        return Panel(grid, title="[bold blue]Session Config[/bold blue]", border_style="blue")

    def get_current_panel(self) -> Panel:
        t_pct = (self.train_progress / self.train_batches * 100) if self.train_batches > 0 else 0
        v_pct = (self.val_progress / self.val_batches * 100) if self.val_batches > 0 else 0
        
        train_bar = Progress(BarColumn(bar_width=None), TextColumn("[bold]{task.percentage:>3.0f}%"), console=console)
        train_bar.add_task("train", total=100, completed=t_pct)
        
        val_bar = Progress(BarColumn(bar_width=None), TextColumn("[bold]{task.percentage:>3.0f}%"), console=console)
        val_bar.add_task("val", total=100, completed=v_pct)

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        
        grid.add_row(
            f"[bold cyan]Epoch:[/bold cyan] {self.current_epoch} / {self.total_epochs}",
            f"[bold yellow]Patience:[/bold yellow] {self.patience_left} / {self.config.get('patience_max', 0)}"
        )
        grid.add_row(f"[bold cyan]Status:[/bold cyan] {self.status}")
        
        content = Group(
            grid,
            Rule(style="dim"),
            Text(f"Training Progress ({self.train_progress}/{self.train_batches})", style="dim"),
            train_bar,
            Text(f"Validation Progress ({self.val_progress}/{self.val_batches})", style="dim"),
            val_bar
        )
        return Panel(content, title="[bold yellow]Execution State[/bold yellow]", border_style="yellow")

    def get_history_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold magenta")
        table.add_column("Epoch", justify="center")
        table.add_column("Score", justify="right", style="green")
        table.add_column("T.Loss", justify="right", style="dim")
        table.add_column("V.Loss", justify="right", style="dim")
        table.add_column("V.AP", justify="right")
        table.add_column("V.R@10", justify="right")
        table.add_column("V.R@15", justify="right")
        table.add_column("V.R@20", justify="right")
        table.add_column("NDCG10", justify="right")
        
        # Show last 15
        for h in self.history[-15:]:
            table.add_row(
                f"{h['epoch']:03d}",
                f"{h['score']:.4f}",
                f"{h['train_loss']:.4f}",
                f"{h['val_loss']:.4f}",
                f"{h['val_ap']:.4f}",
                f"{h['val_r10']:.4f}",
                f"{h['val_r15']:.4f}",
                f"{h['val_r20']:.4f}",
                f"{h['val_ndcg10']:.4f}"
            )
        return Panel(table, title="[bold magenta]Training History[/bold magenta]", border_style="magenta")

    def get_log_panel(self) -> Panel:
        log_content = "\n".join(self.event_log)
        return Panel(log_content, title="[dim]Event Log[/dim]", border_style="dim")

    def render(self) -> Layout:
        l = self.make_layout()
        l["params"].update(self.get_params_panel())
        l["current"].update(self.get_current_panel())
        l["middle"].update(self.get_history_panel())
        l["bottom"].update(self.get_log_panel())
        return l


# ============================================================
# Dataset
# ============================================================
class VRPEdgeDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, "inst_*", "graph_data.pt")))
        if not self.files:
            raise FileNotFoundError(f"No graph_data.pt files found under: {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        obj = torch.load(self.files[idx], map_location="cpu")

        data = Data(
            x=obj["x"].float(),
            edge_index=obj["edge_index"].long(),
            edge_attr=obj["edge_attr"].float(),
            y=obj["y"].float(),
        )

        data.node_ids = obj["node_ids"].long()
        data.node_feature_cols = obj.get("node_feature_cols", [])
        data.edge_feature_cols = obj.get("edge_feature_cols", [])
        data.instance_name = os.path.basename(os.path.dirname(self.files[idx]))
        data.source_file = self.files[idx]
        data.num_pos = int((data.y > 0.5).sum().item())
        data.num_edges = int(data.y.numel())
        return data


# ============================================================
# Split
# ============================================================
def save_split_file(names: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


def load_split_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_or_load_split(dataset: VRPEdgeDataset, split_dir: str, val_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, "train.txt")
    val_path = os.path.join(split_dir, "val.txt")

    all_names = [os.path.basename(os.path.dirname(fp)) for fp in dataset.files]
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_names = load_split_file(train_path)
        val_names = load_split_file(val_path)

        train_idx = [name_to_idx[n] for n in train_names if n in name_to_idx]
        val_idx = [name_to_idx[n] for n in val_names if n in name_to_idx]

        if len(train_idx) > 0 and len(val_idx) > 0:
            return train_idx, val_idx

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_val = max(1, int(round(len(indices) * val_ratio)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if len(train_idx) == 0:
        train_idx = val_idx
        val_idx = val_idx[:1]

    train_names = [all_names[i] for i in train_idx]
    val_names = [all_names[i] for i in val_idx]

    save_split_file(train_names, train_path)
    save_split_file(val_names, val_path)
    return train_idx, val_idx


# ============================================================
# Feature-aware utilities
# ============================================================
def get_feature_index_map(feature_cols: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(feature_cols)}


def safe_col(idx_map: Dict[str, int], name: str) -> Optional[int]:
    return idx_map.get(name, None)


def extract_edge_prior(edge_attr: torch.Tensor, feature_cols: List[str]) -> torch.Tensor:
    """
    Build a lightweight heuristic prior from preprocessing features.
    This helps early training and stabilizes recall at small top-k.
    """
    idx = get_feature_index_map(feature_cols)
    device = edge_attr.device
    prior = torch.zeros(edge_attr.size(0), device=device, dtype=edge_attr.dtype)

    def add(name: str, weight: float, sign: float = 1.0):
        nonlocal prior
        col = safe_col(idx, name)
        if col is not None:
            prior = prior + sign * weight * edge_attr[:, col]

    add("is_time_feasible", 1.50, +1.0)
    add("tw_slack_norm", 0.70, +1.0)
    add("slack_after_service_norm", 0.60, +1.0)
    add("tw_overlap_norm", 0.50, +1.0)
    add("distance_norm", 0.90, -1.0)
    add("route_proxy_cost_norm", 0.70, -1.0)
    add("waiting_time_norm", 0.45, -1.0)
    add("touches_depot", 0.15, +1.0)

    return prior.unsqueeze(-1)


# ============================================================
# Model
# ============================================================
class ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class EdgeRankGNNRefined(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_feature_cols: List[str],
        hidden_dim: int = 160,
        num_layers: int = 5,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feature_cols = list(edge_feature_cols)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
            self.ffns.append(ResidualMLP(hidden_dim, dropout))

        self.src_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dst_gate = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim, hidden_dim)

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        prior = extract_edge_prior(edge_attr, self.edge_feature_cols)
        edge_attr_aug = torch.cat([edge_attr, prior], dim=-1)

        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr_aug)

        for conv, norm, ffn in zip(self.convs, self.norms, self.ffns):
            h = conv(x, edge_index, e)
            h = norm(h)
            h = F.relu(h)
            x = x + h
            x = ffn(x)

        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        gated_interaction = torch.sigmoid(self.src_gate(x_src) + self.dst_gate(x_dst) + self.edge_gate(e))
        pair_mul = gated_interaction * (x_src * x_dst)

        edge_repr = torch.cat([x_src, x_dst, e, pair_mul], dim=-1)
        logits = self.edge_head(edge_repr).squeeze(-1)
        return logits


# ============================================================
# Losses
# ============================================================
def compute_global_pos_weight(graphs: List[Data], device: torch.device) -> torch.Tensor:
    total_pos = 0.0
    total_count = 0.0

    for g in graphs:
        y = g.y.float()
        total_pos += y.sum().item()
        total_count += y.numel()

    total_neg = total_count - total_pos
    if total_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=device)
    return torch.tensor(total_neg / max(total_pos, 1.0), dtype=torch.float32, device=device)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    focal_factor = (1.0 - pt).pow(gamma)
    return (focal_factor * bce).mean()


def pairwise_ranking_loss_single_graph(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_pos: int = 768,
    max_neg: int = 768,
) -> torch.Tensor:
    pos_idx = torch.where(labels > 0.5)[0]
    neg_idx = torch.where(labels <= 0.5)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, device=logits.device)

    if len(pos_idx) > max_pos:
        pos_scores = logits[pos_idx]
        keep = torch.topk(pos_scores, k=max_pos, largest=False).indices
        pos_idx = pos_idx[keep]

    if len(neg_idx) > max_neg:
        neg_scores = logits[neg_idx]
        keep = torch.topk(neg_scores, k=max_neg, largest=True).indices
        neg_idx = neg_idx[keep]

    pos_scores = logits[pos_idx]
    neg_scores = logits[neg_idx]
    diff = pos_scores[:, None] - neg_scores[None, :]
    return F.softplus(-diff).mean()


def topk_recall_surrogate_single_graph(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_frac: float = 0.10,
) -> torch.Tensor:
    """
    Encourage positives to move above the kth score threshold.
    Differentiable enough for practical training.
    """
    pos_idx = torch.where(labels > 0.5)[0]
    if len(pos_idx) == 0:
        return torch.tensor(0.0, device=logits.device)

    n = logits.numel()
    k = max(1, math.ceil(n * top_frac))
    threshold = torch.topk(logits, k=k).values[-1]
    pos_logits = logits[pos_idx]
    return F.softplus(-(pos_logits - threshold)).mean()


def compute_edge_batch(batch: Data) -> torch.Tensor:
    return batch.batch[batch.edge_index[0]]


def compute_graphwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    edge_batch: torch.Tensor,
    global_pos_weight: torch.Tensor,
    rank_weight: float = 0.45,
    topk_weight: float = 0.35,
    focal_gamma: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_graphs = edge_batch.unique(sorted=True)

    cls_losses = []
    rank_losses = []
    topk_losses = []

    for g in unique_graphs:
        mask = edge_batch == g
        g_logits = logits[mask]
        g_labels = labels[mask]

        if g_labels.numel() == 0:
            continue

        cls = focal_bce_with_logits(g_logits, g_labels, pos_weight=global_pos_weight, gamma=focal_gamma)
        rank = pairwise_ranking_loss_single_graph(g_logits, g_labels)
        topk = topk_recall_surrogate_single_graph(g_logits, g_labels, top_frac=0.10)

        cls_losses.append(cls)
        rank_losses.append(rank)
        topk_losses.append(topk)

    if len(cls_losses) == 0:
        zero = torch.tensor(0.0, device=logits.device)
        return zero, zero, zero, zero

    cls = torch.stack(cls_losses).mean()
    rank = torch.stack(rank_losses).mean()
    topk = torch.stack(topk_losses).mean()
    total = cls + rank_weight * rank + topk_weight * topk
    return total, cls.detach(), rank.detach(), topk.detach()


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, frac: float = 0.10) -> float:
    n = scores.numel()
    if n == 0:
        return 0.0
    total_pos = labels.sum().item()
    if total_pos == 0:
        return 0.0

    k = max(1, math.ceil(n * frac))
    topk_idx = torch.topk(scores, k=k).indices
    selected = labels[topk_idx].sum().item()
    return float(selected / total_pos)


@torch.no_grad()
def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, frac: float = 0.10) -> float:
    n = scores.numel()
    if n == 0:
        return 0.0
    k = max(1, math.ceil(n * frac))
    topk_idx = torch.topk(scores, k=k).indices
    selected = labels[topk_idx].sum().item()
    return float(selected / k)


@torch.no_grad()
def average_precision_like(scores: torch.Tensor, labels: torch.Tensor) -> float:
    n = scores.numel()
    if n == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    y = labels[order]
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, len(y) + 1, device=y.device)

    denom = y.sum().item()
    if denom == 0:
        return 0.0
    return float((precision * y).sum().item() / denom)


@torch.no_grad()
def ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, frac: float = 0.10) -> float:
    n = scores.numel()
    if n == 0:
        return 0.0

    k = max(1, math.ceil(n * frac))
    order = torch.argsort(scores, descending=True)[:k]
    gains = labels[order]

    discounts = torch.log2(torch.arange(2, k + 2, device=scores.device).float())
    dcg = (gains / discounts).sum().item()

    ideal = torch.sort(labels, descending=True).values[:k]
    idcg = (ideal / discounts).sum().item()
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


@torch.no_grad()
def avg_positive_rank_percentile(scores: torch.Tensor, labels: torch.Tensor) -> float:
    n = scores.numel()
    if n == 0:
        return 100.0

    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32)

    pos_idx = torch.where(labels > 0.5)[0]
    if len(pos_idx) == 0:
        return 100.0

    pos_ranks = ranks[pos_idx]
    return float((pos_ranks / n).mean().item() * 100.0)


@torch.no_grad()
def evaluate_topk_hit_rates(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    return {
        "r10": recall_at_k(scores, labels, 0.10),
        "r15": recall_at_k(scores, labels, 0.15),
        "r20": recall_at_k(scores, labels, 0.20),
        "p10": precision_at_k(scores, labels, 0.10),
        "ndcg10": ndcg_at_k(scores, labels, 0.10),
    }


# ============================================================
# Training / evaluation
# ============================================================
def mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else 0.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    global_pos_weight: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    rank_weight: float = 0.45,
    topk_weight: float = 0.35,
    focal_gamma: float = 1.5,
    on_batch: Optional[Callable] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_rank = 0.0
    total_topk = 0.0
    batch_count = 0

    graph_ap = []
    graph_r10 = []
    graph_r15 = []
    graph_r20 = []
    graph_p10 = []
    graph_ndcg10 = []
    graph_pos_rank_pct = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labels = batch.y.float()
        edge_batch = compute_edge_batch(batch)

        loss, cls, rank, topk = compute_graphwise_loss(
            logits=logits,
            labels=labels,
            edge_batch=edge_batch,
            global_pos_weight=global_pos_weight,
            rank_weight=rank_weight,
            topk_weight=topk_weight,
            focal_gamma=focal_gamma,
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        scores = torch.sigmoid(logits)
        for g in edge_batch.unique(sorted=True):
            mask = edge_batch == g
            g_scores = scores[mask]
            g_labels = labels[mask]

            topk_metrics = evaluate_topk_hit_rates(g_scores, g_labels)
            graph_ap.append(average_precision_like(g_scores, g_labels))
            graph_r10.append(topk_metrics["r10"])
            graph_r15.append(topk_metrics["r15"])
            graph_r20.append(topk_metrics["r20"])
            graph_p10.append(topk_metrics["p10"])
            graph_ndcg10.append(topk_metrics["ndcg10"])
            graph_pos_rank_pct.append(avg_positive_rank_percentile(g_scores, g_labels))

        total_loss += float(loss.item())
        total_cls += float(cls.item())
        total_rank += float(rank.item())
        total_topk += float(topk.item())
        batch_count += 1
        if on_batch:
            on_batch(batch_count)

    return {
        "loss": total_loss / max(batch_count, 1),
        "cls": total_cls / max(batch_count, 1),
        "rank": total_rank / max(batch_count, 1),
        "topk": total_topk / max(batch_count, 1),
        "ap": mean_or_zero(graph_ap),
        "r10": mean_or_zero(graph_r10),
        "r15": mean_or_zero(graph_r15),
        "r20": mean_or_zero(graph_r20),
        "p10": mean_or_zero(graph_p10),
        "ndcg10": mean_or_zero(graph_ndcg10),
        "avg_pos_rank_pct": mean_or_zero(graph_pos_rank_pct),
    }


# ============================================================
# Ranked edge export
# ============================================================
@torch.no_grad()
def rank_edges_for_instance(model: nn.Module, data: Data, device: torch.device, output_dir: str) -> str:
    model.eval()
    data = data.to(device)

    logits = model(data.x, data.edge_index, data.edge_attr)
    scores = torch.sigmoid(logits).detach().cpu()

    src_idx = data.edge_index[0].detach().cpu()
    dst_idx = data.edge_index[1].detach().cpu()
    node_ids = data.node_ids.detach().cpu()

    from_nodes = node_ids[src_idx].numpy()
    to_nodes = node_ids[dst_idx].numpy()

    df = pd.DataFrame({
        "from": from_nodes,
        "to": to_nodes,
        "score": scores.numpy(),
        "label": data.y.detach().cpu().numpy(),
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{data.instance_name}.csv")
    df.to_csv(out_file, index=False)
    return out_file


def export_ranked_subset(
    model: nn.Module,
    dataset: VRPEdgeDataset,
    indices: List[int],
    subset_name: str,
    device: torch.device,
    ranked_root: str,
) -> None:
    subset_dir = os.path.join(ranked_root, subset_name)
    os.makedirs(subset_dir, exist_ok=True)

    print(f"\nGenerating ranked edge files for {subset_name}...")
    for idx in indices:
        data = dataset[idx]
        out_file = rank_edges_for_instance(model, data, device, subset_dir)
        print(f"  Saved: {out_file}")


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refined GNN trainer for VRPTW edge ranking")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--ranked_dir", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--rank_loss_weight", type=float, default=0.45)
    parser.add_argument("--topk_loss_weight", type=float, default=0.35)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(script_dir, "processed_data")
    model_dir = args.model_dir or os.path.join(script_dir, "models")
    ranked_dir = args.ranked_dir or os.path.join(script_dir, "ranked_output")
    split_dir = args.split_dir or os.path.join(script_dir, "splits")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    set_seed(args.seed)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ranked_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    dataset = VRPEdgeDataset(data_root)
    train_idx, val_idx = make_or_load_split(dataset, split_dir, val_ratio=args.val_ratio)

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    sample = dataset[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    edge_feature_cols = list(sample.edge_feature_cols)

    model = EdgeRankGNNRefined(
        node_dim=node_dim,
        edge_dim=edge_dim,
        edge_feature_cols=edge_feature_cols,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, len(train_loader) * args.warmup_epochs)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    global_pos_weight = compute_global_pos_weight(train_set, device)

    best_score = -1.0
    best_epoch = -1
    patience_left = args.patience

    best_model_path = os.path.join(model_dir, "best_edge_ranker_refined.pt")
    history_path = os.path.join(model_dir, "train_history_refined.json")

    history = []

    # ── HUD Setup ────────────────────────────────────────────────────────────
    hud_config = {
        "device": device,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_graphs": len(train_set),
        "val_graphs": len(val_set),
        "edge_feats": len(edge_feature_cols),
        "patience_max": args.patience,
        "weights": f"R:{args.rank_loss_weight} T:{args.topk_loss_weight} G:{args.focal_gamma}"
    }
    hud = TrainingHUD(
        total_epochs=args.epochs,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        config=hud_config
    ) if RICH_AVAILABLE else None

    if not RICH_AVAILABLE:
        print("=" * 84)
        print("REFINED GNN TRAINER FOR VRPTW EDGE RANKING")
        print("SCRIPT PATH:", os.path.abspath(__file__))
        print("=" * 84)
        print(f"Device: {device}")
        print(f"Train graphs: {len(train_set)} | Val graphs: {len(val_set)}")
        print(f"Node dim: {node_dim} | Edge dim: {edge_dim}")
        print(f"Global pos_weight: {global_pos_weight.item():.4f}")
        print(f"Edge features: {edge_feature_cols}")

    # ── Live Execution ───────────────────────────────────────────────────────
    live_ctx = Live(hud.render(), console=console, refresh_per_second=4, screen=False) if RICH_AVAILABLE else None
    if live_ctx: 
        live_ctx.start()
        hud.log("Training session started")

    try:
        for epoch in range(1, args.epochs + 1):
            if hud:
                hud.current_epoch = epoch
                hud.status = "Training..."
                hud.train_progress = 0
                hud.val_progress = 0
                hud.lr = optimizer.param_groups[0]["lr"]
                live_ctx.update(hud.render())

            def on_train_batch(count):
                if hud:
                    hud.train_progress = count
                    live_ctx.update(hud.render())

            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                global_pos_weight=global_pos_weight,
                optimizer=optimizer,
                scheduler=scheduler,
                rank_weight=args.rank_loss_weight,
                topk_weight=args.topk_loss_weight,
                focal_gamma=args.focal_gamma,
                on_batch=on_train_batch
            )

            if hud:
                hud.status = "Validating..."
                live_ctx.update(hud.render())

            def on_val_batch(count):
                if hud:
                    hud.val_progress = count
                    live_ctx.update(hud.render())

            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                global_pos_weight=global_pos_weight,
                optimizer=None,
                scheduler=None,
                rank_weight=args.rank_loss_weight,
                topk_weight=args.topk_loss_weight,
                focal_gamma=args.focal_gamma,
                on_batch=on_val_batch
            )

            val_score = (
                0.30 * val_metrics["ap"] +
                0.25 * val_metrics["r10"] +
                0.20 * val_metrics["r15"] +
                0.10 * val_metrics["p10"] +
                0.10 * val_metrics["ndcg10"] +
                0.05 * (1.0 - val_metrics["avg_pos_rank_pct"] / 100.0)
            )

            epoch_record = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "val": val_metrics,
                "val_score": float(val_score),
            }
            history.append(epoch_record)

            if hud:
                hud.history.append({
                    "epoch": epoch,
                    "score": val_score,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_ap": val_metrics["ap"],
                    "val_r10": val_metrics["r10"],
                    "val_r15": val_metrics["r15"],
                    "val_r20": val_metrics["r20"],
                    "val_ndcg10": val_metrics["ndcg10"],
                })
                hud.status = "Epoch Summary"
                live_ctx.update(hud.render())
            else:
                print(
                    f"Epoch {epoch:03d} | Train Loss {train_metrics['loss']:.4f} | "
                    f"Val Loss {val_metrics['loss']:.4f} | Score {val_score:.4f}"
                )

            if val_score > best_score:
                best_score = float(val_score)
                best_epoch = epoch
                patience_left = args.patience
                
                if hud:
                    hud.best_score = best_score
                    hud.patience_left = patience_left
                    hud.log(f"[bold green]Best Score![/bold green] ({best_score:.4f})")
                
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "node_dim": node_dim,
                        "edge_dim": edge_dim,
                        "edge_feature_cols": edge_feature_cols,
                        "args": vars(args),
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                    },
                    best_model_path,
                )
                if hud: hud.log(f"Model saved to models/")
            else:
                patience_left -= 1
                if hud: 
                    hud.patience_left = patience_left
                    hud.log(f"No improvement (Patience: {patience_left})")

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if patience_left <= 0:
                if hud: hud.log("[red]Early stopping triggered[/red]")
                break

    finally:
        if live_ctx:
            hud.status = "Finished"
            live_ctx.update(hud.render())
            live_ctx.stop()

    print(f"\nBest validation score: {best_score:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    export_ranked_subset(model, dataset, train_idx, "train", device, ranked_dir)
    export_ranked_subset(model, dataset, val_idx, "val", device, ranked_dir)


if __name__ == "__main__":
    main()
