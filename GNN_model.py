import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
import random

# ##############################################################################
# Cell 1: Imports
# ##############################################################################

# ##############################################################################
# Cell 2: PyG Imports
# ##############################################################################

# ##############################################################################
# Cell 4: Utilities
# ##############################################################################

def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_id_mapping(nodes_df: pd.DataFrame, node_id_col: str = "node_id"):
    node_ids = nodes_df[node_id_col].tolist()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    return id2idx

def infer_node_feature_cols(nodes_df: pd.DataFrame, node_id_col: str = "node_id"):
    """
    Infer node features from the uploaded node_features.csv.
    Excludes only identifier columns.
    """
    exclude = {node_id_col}
    cols = [c for c in nodes_df.columns if c not in exclude]
    if not cols:
        raise ValueError("No usable node feature columns found in node_features.csv")
    return cols

def infer_edge_feature_cols(edge_df: pd.DataFrame):
    """
    Infer edge features from the uploaded edge_features.csv.
    Avoid label leakage by excluding key columns and target columns.
    """
    exclude = {"from", "to", "label", "y", "in_solution"}
    cols = [c for c in edge_df.columns if c not in exclude]
    if not cols:
        raise ValueError("No usable edge feature columns found in edge_features.csv")
    return cols

def standardize_numeric_df(df: pd.DataFrame, binary_cols=None, eps: float = 1e-8):
    """
    Standardize non-binary numeric columns; leave binary indicator columns unchanged.
    Returns standardized dataframe plus per-column statistics.
    """
    out = df.copy()
    binary_cols = set(binary_cols or [])
    stats = {}
    for c in out.columns:
        vals = out[c].astype(float)
        unique_vals = set(pd.Series(vals).dropna().unique().tolist())
        is_binary = c in binary_cols or unique_vals.issubset({0.0, 1.0})
        if is_binary:
            stats[c] = {"mean": 0.0, "std": 1.0, "binary": True}
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if std < eps:
            std = 1.0
        out[c] = (vals - mean) / std
        stats[c] = {"mean": mean, "std": std, "binary": False}
    return out, stats

def check_edge_alignment(ei_df, ef_df, y_df):
    """
    Ensure edge_index, edge_features, y_label contain the same edges in the same order.
    If not aligned, we will align by merging on (from,to).
    """
    ei_pairs = list(zip(ei_df["from"].tolist(), ei_df["to"].tolist()))
    ef_pairs = list(zip(ef_df["from"].tolist(), ef_df["to"].tolist()))
    y_pairs  = list(zip(y_df["from"].tolist(),  y_df["to"].tolist()))
    aligned = (ei_pairs == ef_pairs) and (ei_pairs == y_pairs)
    return aligned

def align_edges(ei_df, ef_df, y_df):
    """
    Align edge_features and y_label to edge_index order using (from,to) keys.
    """
    key_cols = ["from", "to"]
    ef_aligned = ei_df.merge(ef_df, on=key_cols, how="left", validate="one_to_one")
    y_aligned  = ei_df.merge(y_df,  on=key_cols, how="left", validate="one_to_one")
    if ef_aligned.isna().any().any():
        missing = ef_aligned[ef_aligned.isna().any(axis=1)][key_cols].head(10)
        raise ValueError(f"Missing edge_features for some edges. Example missing keys:\n{missing}")
    if y_aligned["label"].isna().any():
        missing = y_aligned[y_aligned["label"].isna()][key_cols].head(10)
        raise ValueError(f"Missing y_label for some edges. Example missing keys:\n{missing}")
    return ef_aligned, y_aligned

# ##############################################################################
# Cell 9: Load Data
# ##############################################################################

def load_vrptw_instance_from_csv(
    node_csv: str,
    edge_index_csv: str,
    edge_feat_csv: str,
    y_csv: str,
    node_feature_cols=None,
    edge_feature_cols=None,
    node_id_col="node_id",
    standardize_features: bool = True,
):
    nodes = pd.read_csv(node_csv)
    ei = pd.read_csv(edge_index_csv)
    ef = pd.read_csv(edge_feat_csv)
    y  = pd.read_csv(y_csv)
    if node_feature_cols is None:
        node_feature_cols = infer_node_feature_cols(nodes, node_id_col=node_id_col)
    if edge_feature_cols is None:
        edge_feature_cols = infer_edge_feature_cols(ef)
    if not check_edge_alignment(ei, ef, y):
        ef_aligned, y_aligned = align_edges(ei, ef, y)
        ef = ef_aligned
        y = y_aligned
    if "in_solution" in edge_feature_cols:
        raise ValueError("edge_feature_cols must not include 'in_solution' because it leaks the label.")
    if len(ei) != len(ef) or len(ei) != len(y):
        raise ValueError("edge_index, edge_features, and y_label must have the same number of rows after alignment.")
    id2idx = build_id_mapping(nodes, node_id_col=node_id_col)
    node_feat_df = nodes[node_feature_cols].copy()
    edge_feat_df = ef[edge_feature_cols].copy()
    if standardize_features:
        node_binary_cols = [c for c in node_feat_df.columns if "depot" in c.lower()]
        edge_binary_cols = [
            c for c in edge_feat_df.columns
            if any(k in c.lower() for k in ["is_", "has_", "touches_depot", "depot"])
        ]
        node_feat_df, node_stats = standardize_numeric_df(node_feat_df, binary_cols=node_binary_cols)
        edge_feat_df, edge_stats = standardize_numeric_df(edge_feat_df, binary_cols=edge_binary_cols)
    else:
        node_stats, edge_stats = {}, {}
    x = torch.tensor(node_feat_df.to_numpy(), dtype=torch.float)
    src = ei["from"].map(id2idx).to_numpy()
    dst = ei["to"].map(id2idx).to_numpy()
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(edge_feat_df.to_numpy(), dtype=torch.float)
    label_col = "label" if "label" in y.columns else y.columns[-1]
    labels = torch.tensor(y[label_col].to_numpy(), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    data.num_nodes = x.size(0)
    data.node_id_order = nodes[node_id_col].tolist()
    data.node_feature_cols = list(node_feature_cols)
    data.edge_feature_cols = list(edge_feature_cols)
    data.node_feature_stats = node_stats
    data.edge_feature_stats = edge_stats
    data.edge_keys = ei[["from", "to"]].copy()
    return data

# ##############################################################################
# Cell 11: Model
# ##############################################################################

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class VRPTWEdgeGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_edge))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res
        row, col = edge_index
        src_h = x[row]
        dst_h = x[col]
        edge_feat = torch.cat([
            src_h,
            dst_h,
            torch.abs(src_h - dst_h),
            edge_attr,
        ], dim=1)
        scores = self.edge_mlp(edge_feat).squeeze(-1)
        return scores

# ##############################################################################
# Cell 14: Loss and Metrics
# ##############################################################################

def bce_with_pos_weight_from_labels(labels: torch.Tensor):
    pos = (labels == 1).sum().clamp(min=1)
    neg = (labels == 0).sum()
    return (neg.float() / pos.float()).clamp(min=1.0)

def listwise_edge_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
):
    pos_mask = labels > 0.5
    if pos_mask.sum() == 0:
        return scores.new_tensor(0.0)
    pred_log_probs = F.log_softmax(scores / temperature, dim=0)
    target = labels.float().clone()
    target = target / target.sum().clamp(min=1.0)
    return -(target * pred_log_probs).sum()

@torch.no_grad()
def compute_soft_pruning_metrics(scores: torch.Tensor, labels: torch.Tensor):
    probs = torch.sigmoid(scores).cpu().numpy()
    labels_np = labels.cpu().numpy()
    pos_mask = labels_np == 1
    neg_mask = labels_np == 0
    n_pos = int(pos_mask.sum())
    n_total = len(labels_np)
    if n_pos == 0 or n_pos == n_total:
        return {"error": "Need both positive and negative examples"}
    sorted_idx = np.argsort(-probs)
    sorted_labels = labels_np[sorted_idx]
    pos_probs = probs[pos_mask]
    neg_probs = probs[neg_mask]
    auc = 0.0
    for pp in pos_probs:
        auc += (neg_probs < pp).sum() + 0.5 * (neg_probs == pp).sum()
    auc /= max(len(pos_probs) * len(neg_probs), 1)
    cum_pos = np.cumsum(sorted_labels)
    ranks = np.arange(1, n_total + 1)
    precision_at_k = cum_pos / ranks
    ap = float((precision_at_k * sorted_labels).sum() / max(n_pos, 1))
    rank_positions = np.empty(n_total, dtype=np.int64)
    rank_positions[sorted_idx] = np.arange(1, n_total + 1)
    pos_ranks = rank_positions[pos_mask]
    discounts = 1.0 / np.log2(np.arange(2, n_total + 2))
    dcg = float((sorted_labels * discounts).sum())
    ideal_labels = np.sort(labels_np)[::-1]
    idcg = float((ideal_labels * discounts).sum())
    ndcg = dcg / max(idcg, 1e-12)
    recall_at_k = {}
    for k_pct in [0.05, 0.10, 0.20, 0.30, 0.50]:
        k = max(1, int(np.ceil(k_pct * n_total)))
        recall_at_k[f"recall@{int(k_pct*100)}%"] = float(sorted_labels[:k].sum() / n_pos)
    min_pos_prob = float(pos_probs.min())
    edges_at_recall1 = int((probs >= (min_pos_prob - 1e-3)).sum())
    return {
        "auc_roc": float(auc),
        "avg_precision": ap,
        "ndcg": float(ndcg),
        "avg_pos_rank": float(pos_ranks.mean()),
        "median_pos_rank": float(np.median(pos_ranks)),
        "avg_pos_rank_pct": float(pos_ranks.mean() / n_total * 100.0),
        "edges_at_recall1": edges_at_recall1,
        "edges_at_recall1_pct": float(edges_at_recall1 / n_total * 100.0),
        "min_pos_prob": min_pos_prob,
        **recall_at_k,
    }

def focal_bce_with_logits(
    scores: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
):
    bce = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(scores)
    pt = probs * labels + (1.0 - probs) * (1.0 - labels)
    focal = (1.0 - pt).pow(gamma)
    return (focal * bce).mean()

def pairwise_margin_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    max_neg_per_pos: int = 5,
):
    pos_scores = scores[labels > 0.5]
    neg_scores = scores[labels <= 0.5]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return scores.new_tensor(0.0)
    num_hard_negs = min(neg_scores.numel(), max(pos_scores.numel() * max_neg_per_pos, 1))
    hard_neg_scores = torch.topk(neg_scores, k=num_hard_negs, largest=True).values
    diff = pos_scores.unsqueeze(1) - hard_neg_scores.unsqueeze(0)
    return F.relu(margin - diff).mean()

def topk_bce_loss(scores: torch.Tensor, labels: torch.Tensor, top_frac: float = 0.2):
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    k = max(1, int(scores.numel() * top_frac))
    top_idx = torch.topk(scores, k=k, largest=True).indices
    pos_weight = bce_with_pos_weight_from_labels(labels)
    return F.binary_cross_entropy_with_logits(scores[top_idx], labels[top_idx], pos_weight=pos_weight)

def get_loss_fn(
    labels: torch.Tensor,
    focal_lambda: float = 0.20,
    listwise_lambda: float = 0.20,
    pairwise_lambda: float = 0.40,
    topk_lambda: float = 0.20,
    margin: float = 0.2,
    max_neg_per_pos: int = 5,
    temperature: float = 0.7,
    focal_gamma: float = 2.0,
    top_frac: float = 0.2,
):
    pos_weight = bce_with_pos_weight_from_labels(labels)
    def loss_fn(scores: torch.Tensor, targets: torch.Tensor):
        focal_bce = focal_bce_with_logits(scores, targets, pos_weight=pos_weight, gamma=focal_gamma)
        listwise = listwise_edge_ranking_loss(scores, targets, temperature=temperature)
        pairwise = pairwise_margin_ranking_loss(
            scores,
            targets,
            margin=margin,
            max_neg_per_pos=max_neg_per_pos,
        )
        topk = topk_bce_loss(scores, targets, top_frac=top_frac)
        total = (
            focal_lambda * focal_bce
            + listwise_lambda * listwise
            + pairwise_lambda * pairwise
            + topk_lambda * topk
        )
        return total, {
            "bce": float(focal_bce.detach().cpu()),
            "listwise": float(listwise.detach().cpu()),
            "pairwise": float(pairwise.detach().cpu()),
            "topk": float(topk.detach().cpu()),
        }
    return loss_fn

# ##############################################################################
# Cell 18: Train/Eval
# ##############################################################################

def train_one_epoch_with_soft_metrics(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_listwise = 0.0
    total_pairwise = 0.0
    total_topk = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(data)
        loss_fn = get_loss_fn(data.y)
        loss, loss_parts = loss_fn(scores, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_bce += loss_parts["bce"]
        total_listwise += loss_parts["listwise"]
        total_pairwise += loss_parts["pairwise"]
        total_topk += loss_parts["topk"]
    denom = max(len(loader), 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "listwise": total_listwise / denom,
        "pairwise": total_pairwise / denom,
        "topk": total_topk / denom,
    }

@torch.no_grad()
def evaluate_soft_pruning(model, loader, device):
    model.eval()
    all_metrics = []
    for data in loader:
        data = data.to(device)
        scores = model(data)
        m = compute_soft_pruning_metrics(scores, data.y)
        if "error" not in m:
            all_metrics.append(m)
    if not all_metrics:
        return {}
    def avg(key):
        vals = [m[key] for m in all_metrics if key in m]
        return float(sum(vals) / len(vals)) if vals else 0.0
    return {k: avg(k) for k in all_metrics[0].keys()}

# ##############################################################################
# Cell 21: Main
# ##############################################################################

def discover_instance_dirs(root_dir: str):
    root_dir = os.path.expanduser(root_dir)
    if not os.path.isdir(root_dir):
        return []
    candidates = glob.glob(os.path.join(root_dir, "*", "node_features.csv")) +                  glob.glob(os.path.join(root_dir, "*", "*", "node_features.csv"))
    inst_dirs = []
    for nf in candidates:
        d = os.path.dirname(nf)
        needed = [
            os.path.join(d, "node_features.csv"),
            os.path.join(d, "edge_index.csv"),
            os.path.join(d, "edge_features.csv"),
            os.path.join(d, "y_label.csv"),
        ]
        if all(os.path.isfile(p) for p in needed):
            inst_dirs.append(d)
    return sorted(list(dict.fromkeys(inst_dirs)))

def maybe_add_current_directory_instance(instance_dirs: list):
    needed = ["node_features.csv", "edge_index.csv", "edge_features.csv", "y_label.csv"]
    if all(os.path.isfile(p) for p in needed):
        cwd = os.getcwd()
        if cwd not in instance_dirs:
            instance_dirs = list(instance_dirs) + [cwd]
    return instance_dirs

def main_soft_pruning():
    seed_all(42)

    # Fix: Construct absolute path to processed_data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    INSTANCE_ROOT = os.path.join(script_dir, "processed_data")

    instance_dirs = discover_instance_dirs(INSTANCE_ROOT)
    instance_dirs = maybe_add_current_directory_instance(instance_dirs)
    if not instance_dirs:
        print(f"No instances found in '{INSTANCE_ROOT}'. Put CSVs in processed_data/<instance>/ or in the current directory.")
        return None, None
    dataset = []
    for d in instance_dirs:
        data_i = load_vrptw_instance_from_csv(
            node_csv=os.path.join(d, "node_features.csv"),
            edge_index_csv=os.path.join(d, "edge_index.csv"),
            edge_feat_csv=os.path.join(d, "edge_features.csv"),
            y_csv=os.path.join(d, "y_label.csv"),
            node_feature_cols=[
                "x", "y", "demand", "ready_time", "due_date", "service_time", "is_depot"
            ],
            edge_feature_cols=[
                c for c in pd.read_csv(os.path.join(d, "edge_features.csv"), nrows=1).columns
                if c not in {"from", "to", "in_solution", "label", "y"}
            ],
            node_id_col="node_id",
            standardize_features=True,
        )
        dataset.append(data_i)
    print(f"Loaded {len(dataset)} instances")
    print("Node features:", dataset[0].node_feature_cols)
    print("Edge features:", dataset[0].edge_feature_cols)
    if len(dataset) >= 5:
        split = max(1, int(0.8 * len(dataset)))
        train_set = dataset[:split]
        val_set = dataset[split:]
        if len(val_set) == 0:
            val_set = train_set
    else:
        train_set = dataset
        val_set = dataset
    train_loader = DataLoader(train_set, batch_size=min(4, len(train_set)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=min(4, len(val_set)), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = dataset[0].x.size(1)
    edge_dim = dataset[0].edge_attr.size(1)
    model = VRPTWEdgeGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=20,
    )
    epochs = 30
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    patience = 60
    no_improve = 0
    print("\n" + "="*146)
    print("SCORING-BASED TRAINING - Dataset-aware edge importance learning")
    print("="*146)
    print(
        f"{'Epoch':>6} | {'Loss':>8} | {'Focal':>8} | {'List':>8} | {'Pair':>8} | {'TopK':>8} | "
        f"{'AUC':>6} | {'AP':>6} | {'NDCG':>6} | {'AvgRank%':>8} | {'R@10%':>6} | {'R@20%':>6}"
    )
    print("-"*146)
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch_with_soft_metrics(model, train_loader, optimizer, device)
        val_metrics = evaluate_soft_pruning(model, val_loader, device)
        selection_score = (
            0.35 * val_metrics.get("recall@10%", 0.0)
            + 0.25 * val_metrics.get("recall@20%", 0.0)
            + 0.20 * val_metrics.get("avg_precision", 0.0)
            + 0.20 * val_metrics.get("ndcg", 0.0)
        )
        scheduler.step(selection_score)
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:6d} | "
                f"{train_stats['loss']:8.4f} | "
                f"{train_stats['bce']:8.4f} | "
                f"{train_stats['listwise']:8.4f} | "
                f"{train_stats['pairwise']:8.4f} | "
                f"{train_stats['topk']:8.4f} | "
                f"{val_metrics.get('auc_roc', 0):6.3f} | "
                f"{val_metrics.get('avg_precision', 0):6.3f} | "
                f"{val_metrics.get('ndcg', 0):6.3f} | "
                f"{val_metrics.get('avg_pos_rank_pct', 0):8.2f} | "
                f"{val_metrics.get('recall@10%', 0):6.3f} | "
                f"{val_metrics.get('recall@20%', 0):6.3f}"
            )
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print("-"*146)
    print(f"Best selection score: {best_score:.4f} at epoch {best_epoch}")
    print("="*146)
    print("\n" + "="*72)
    print("FINAL SCORING RESULTS")
    print("="*72)
    final_metrics = evaluate_soft_pruning(model, val_loader, device)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    return model, dataset

# ##############################################################################
# Cell 23: Save Pruned Graph
# ##############################################################################

@torch.no_grad()
def soft_prune_edges(model, data, device):
    model.eval()
    data = data.to(device)
    scores = model(data)
    edge_scores = scores.cpu().numpy().flatten()
    edge_probs = torch.sigmoid(scores).cpu().numpy().flatten()
    rank_of_edge = np.empty_like(edge_probs, dtype=np.int64)
    rank_of_edge[np.argsort(-edge_scores)] = np.arange(1, len(edge_scores) + 1)
    return edge_scores, edge_probs, rank_of_edge

@torch.no_grad()
def find_recall_1_threshold(model, data, device, safety_margin=0.01):
    model.eval()
    data = data.to(device)
    scores = model(data)
    edge_probs = torch.sigmoid(scores).cpu().numpy().flatten()
    labels = data.y.cpu().numpy().flatten()
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() == 0:
        return 0.0, 0.0, {"error": "No positive edges found"}
    pos_probs = edge_probs[pos_mask]
    neg_probs = edge_probs[neg_mask]
    min_pos_prob = pos_probs.min()
    threshold = max(0.0, float(min_pos_prob - safety_margin))
    edges_at_recall1 = int((edge_probs >= threshold).sum())
    stats = {
        "min_pos_prob": float(min_pos_prob),
        "max_pos_prob": float(pos_probs.max()),
        "mean_pos_prob": float(pos_probs.mean()),
        "max_neg_prob": float(neg_probs.max()) if neg_mask.sum() > 0 else 0.0,
        "mean_neg_prob": float(neg_probs.mean()) if neg_mask.sum() > 0 else 0.0,
        "num_positive": int(pos_mask.sum()),
        "num_negative": int(neg_mask.sum()),
        "edges_at_recall1": edges_at_recall1,
        "edges_at_recall1_pct": float(edges_at_recall1 / len(edge_probs) * 100.0),
    }
    return threshold, float(min_pos_prob), stats

def build_safe_keep_mask(
    nodes_df: pd.DataFrame,
    edge_index_df: pd.DataFrame,
    edge_scores: np.ndarray,
    edge_probs: np.ndarray,
    edge_features_df: pd.DataFrame = None,
    global_keep_ratio: float = 0.03,
    min_global_keep: int = 220,
    max_global_keep: int = 900,
    per_node_out: int = 3,
    per_node_in: int = 3,
    keep_all_depot_touches: bool = True,
    depot_extra_topk: int = 30,
):
    n_edges = len(edge_index_df)
    keep_mask = np.zeros(n_edges, dtype=bool)
    global_budget = int(round(global_keep_ratio * n_edges))
    global_budget = max(min_global_keep, global_budget)
    global_budget = min(max_global_keep, global_budget, n_edges)
    top_global_idx = np.argsort(-edge_scores)[:global_budget]
    keep_mask[top_global_idx] = True
    work_df = edge_index_df.copy()
    work_df["_row_id"] = np.arange(n_edges)
    work_df["_score"] = edge_scores
    work_df["_prob"] = edge_probs
    if per_node_out > 0:
        for node_id, group in work_df.groupby("from"):
            chosen = group.nlargest(min(per_node_out, len(group)), "_score")["_row_id"].values
            keep_mask[chosen] = True
    if per_node_in > 0:
        for node_id, group in work_df.groupby("to"):
            chosen = group.nlargest(min(per_node_in, len(group)), "_score")["_row_id"].values
            keep_mask[chosen] = True
    depot_ids = []
    if "is_depot" in nodes_df.columns:
        depot_ids = nodes_df.loc[nodes_df["is_depot"] == 1, "node_id"].tolist() if "node_id" in nodes_df.columns else nodes_df.index[nodes_df["is_depot"] == 1].tolist()
    if not depot_ids:
        depot_ids = [0]
    if keep_all_depot_touches:
        depot_touch_mask = work_df["from"].isin(depot_ids) | work_df["to"].isin(depot_ids)
        keep_mask[depot_touch_mask.values] = True
    if depot_extra_topk > 0:
        depot_out = work_df[work_df["from"].isin(depot_ids)]
        depot_in = work_df[work_df["to"].isin(depot_ids)]
        if len(depot_out) > 0:
            chosen = depot_out.nlargest(min(depot_extra_topk, len(depot_out)), "_score")["_row_id"].values
            keep_mask[chosen] = True
        if len(depot_in) > 0:
            chosen = depot_in.nlargest(min(depot_extra_topk, len(depot_in)), "_score")["_row_id"].values
            keep_mask[chosen] = True
    return keep_mask

def summarize_pruning_from_labels(keep_mask: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels).astype(int)
    keep_mask = np.asarray(keep_mask).astype(bool)
    pos_total = int((labels == 1).sum())
    neg_total = int((labels == 0).sum())
    kept_total = int(keep_mask.sum())
    pruned_total = int(len(labels) - kept_total)
    if pos_total > 0:
        kept_pos = int(((labels == 1) & keep_mask).sum())
        pos_recall = kept_pos / pos_total
    else:
        kept_pos = 0
        pos_recall = 1.0
    if neg_total > 0:
        pruned_neg = int(((labels == 0) & (~keep_mask)).sum())
        neg_prune_rate = pruned_neg / neg_total
    else:
        pruned_neg = 0
        neg_prune_rate = 0.0
    return {
        "kept_edges": kept_total,
        "pruned_edges": pruned_total,
        "kept_edges_pct": float(kept_total / len(labels) * 100.0),
        "pruned_edges_pct": float(pruned_total / len(labels) * 100.0),
        "positive_edges": pos_total,
        "kept_positive_edges": kept_pos,
        "positive_recall": float(pos_recall),
        "pruned_negative_edges": pruned_neg,
        "negative_prune_rate": float(neg_prune_rate),
    }

def save_safe_pruned_graph_for_gurobi(
    original_node_csv: str,
    original_edge_index_csv: str,
    original_edge_features_csv: str,
    edge_scores: np.ndarray,
    edge_probs: np.ndarray,
    keep_mask: np.ndarray,
    output_dir: str,
    instance_name: str = "safe_pruned_instance"
):
    os.makedirs(output_dir, exist_ok=True)
    nodes_df = pd.read_csv(original_node_csv)
    edge_index_df = pd.read_csv(original_edge_index_csv)
    edge_features_df = pd.read_csv(original_edge_features_csv)
    edges_df = edge_index_df.copy()
    edges_df["edge_score"] = edge_scores
    edges_df["prob"] = edge_probs
    edges_df["rank"] = np.argsort(np.argsort(-edge_scores)) + 1
    edges_df["priority_score"] = (edge_scores - edge_scores.min()) / (edge_scores.max() - edge_scores.min() + 1e-8)
    edges_df["keep_for_solver"] = keep_mask.astype(int)
    if len(edge_features_df) == len(edges_df):
        for col in edge_features_df.columns:
            if col not in ["from", "to"]:
                edges_df[col] = edge_features_df[col].values
    kept_edges_df = edges_df[edges_df["keep_for_solver"] == 1].copy().reset_index(drop=True)
    node_output = os.path.join(output_dir, f"{instance_name}_nodes.csv")
    full_edge_output = os.path.join(output_dir, f"{instance_name}_edges_scored_full.csv")
    pruned_edge_output = os.path.join(output_dir, f"{instance_name}_edges_solver.csv")
    nodes_df.to_csv(node_output, index=False)
    edges_df.to_csv(full_edge_output, index=False)
    kept_edges_df.to_csv(pruned_edge_output, index=False)
    stats = {
        "nodes_file": node_output,
        "full_edges_file": full_edge_output,
        "solver_edges_file": pruned_edge_output,
        "num_nodes": len(nodes_df),
        "num_edges_full": len(edges_df),
        "num_edges_solver": len(kept_edges_df),
        "solver_keep_pct": float(len(kept_edges_df) / max(len(edges_df), 1) * 100.0),
        "score_min": float(edge_scores.min()),
        "score_max": float(edge_scores.max()),
        "score_mean": float(edge_scores.mean()),
        "score_std": float(edge_scores.std()),
        "prob_min": float(edge_probs.min()),
        "prob_max": float(edge_probs.max()),
        "prob_mean": float(edge_probs.mean()),
        "prob_std": float(edge_probs.std()),
    }
    print(f"✓ Saved scored full graph to : {full_edge_output}")
    print(f"✓ Saved solver edge set to   : {pruned_edge_output}")
    print(f"  - Nodes: {stats['num_nodes']}")
    print(f"  - Full edges : {stats['num_edges_full']}")
    print(f"  - Solver edges: {stats['num_edges_solver']} ({stats['solver_keep_pct']:.2f}% kept)")
    print(f"  - Score range: [{stats['score_min']:.4f}, {stats['score_max']:.4f}]")
    print(f"  - Prob range : [{stats['prob_min']:.4f}, {stats['prob_max']:.4f}]")
    return stats

def batch_safe_prune_and_save(
    model,
    instance_dirs: list,
    output_base_dir: str = "safe_pruned_graphs",
    device=None,
    global_keep_ratio: float = 0.03,
    min_global_keep: int = 220,
    max_global_keep: int = 900,
    per_node_out: int = 3,
    per_node_in: int = 3,
    keep_all_depot_touches: bool = True,
    depot_extra_topk: int = 30,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for inst_dir in instance_dirs:
        inst_name = os.path.basename(inst_dir)
        print(f"{'='*60}")
        print(f"Scoring + safe pruning: {inst_name}")
        print(f"{'='*60}")
        try:
            node_csv = os.path.join(inst_dir, "node_features.csv")
            edge_index_csv = os.path.join(inst_dir, "edge_index.csv")
            edge_feat_csv = os.path.join(inst_dir, "edge_features.csv")
            y_csv = os.path.join(inst_dir, "y_label.csv")
            data = load_vrptw_instance_from_csv(
                node_csv=node_csv,
                edge_index_csv=edge_index_csv,
                edge_feat_csv=edge_feat_csv,
                y_csv=y_csv,
            )
            nodes_df = pd.read_csv(node_csv)
            edge_index_df = pd.read_csv(edge_index_csv)
            edge_features_df = pd.read_csv(edge_feat_csv)
            y_df = pd.read_csv(y_csv)
            edge_scores, edge_probs, edge_ranks = soft_prune_edges(model, data, device)
            keep_mask = build_safe_keep_mask(
                nodes_df=nodes_df,
                edge_index_df=edge_index_df,
                edge_scores=edge_scores,
                edge_probs=edge_probs,
                edge_features_df=edge_features_df,
                global_keep_ratio=global_keep_ratio,
                min_global_keep=min_global_keep,
                max_global_keep=max_global_keep,
                per_node_out=per_node_out,
                per_node_in=per_node_in,
                keep_all_depot_touches=keep_all_depot_touches,
                depot_extra_topk=depot_extra_topk,
            )
            output_dir = os.path.join(output_base_dir, inst_name)
            result = save_safe_pruned_graph_for_gurobi(
                original_node_csv=node_csv,
                original_edge_index_csv=edge_index_csv,
                original_edge_features_csv=edge_feat_csv,
                edge_scores=edge_scores,
                edge_probs=edge_probs,
                keep_mask=keep_mask,
                output_dir=output_dir,
                instance_name=inst_name,
            )
            if "y" in y_df.columns:
                labels = y_df["y"].values
            elif "label" in y_df.columns:
                labels = y_df["label"].values
            elif "in_solution" in y_df.columns:
                labels = y_df["in_solution"].values
            else:
                label_col = [c for c in y_df.columns if c not in ["from", "to"]][0]
                labels = y_df[label_col].values
            prune_stats = summarize_pruning_from_labels(keep_mask, labels)
            result.update(prune_stats)
            result["instance_name"] = inst_name
            result["global_keep_ratio"] = global_keep_ratio
            result["per_node_out"] = per_node_out
            result["per_node_in"] = per_node_in
            results.append(result)
            print(f"  Positive recall after safe pruning : {prune_stats['positive_recall']:.4f}")
            print(f"  Negative prune rate                : {prune_stats['negative_prune_rate']:.4f}")
            print(f"  Kept edges                         : {prune_stats['kept_edges']} / {len(labels)}")
        except Exception as e:
            print(f"✗ Failed on {inst_name}: {e}")
            import traceback
            traceback.print_exc()
    print(f"{'='*60}")
    print(f"{'='*60}")
    print(f"Completed safe-pruning export for {len(results)}/{len(instance_dirs)} instances")
    print(f"{'='*60}")
    return results

def save_single_scored_instance(model, data, original_node_csv, original_edge_index_csv, original_edge_features_csv,
                                output_dir="single_scored", instance_name="instance", device='cpu'):
    edge_scores, edge_probs, edge_ranks = soft_prune_edges(model, data, device=device)
    return save_soft_pruned_graph_for_gurobi(
        original_node_csv=original_node_csv,
        original_edge_index_csv=original_edge_index_csv,
        original_edge_features_csv=original_edge_features_csv,
        edge_scores=edge_scores,
        edge_probs=edge_probs,
        output_dir=output_dir,
        instance_name=instance_name,
    )

# ##############################################################################
# Cell 28: Complete Workflow
# ##############################################################################

if __name__ == "__main__":
    try:
        model, dataset = main_soft_pruning()
        if model is not None:
            # Use absolute path to processed_data relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            instance_root = os.path.join(script_dir, "processed_data")
            instance_dirs = discover_instance_dirs(instance_root)
            
            # Use absolute path for output directory as well
            output_base_dir = os.path.join(script_dir, "safe_pruned_graphs")
            
            results = batch_safe_prune_and_save(
                model=model,
                instance_dirs=instance_dirs,
                output_base_dir=output_base_dir,
                global_keep_ratio=0.03,
                min_global_keep=220,
                max_global_keep=900,
                per_node_out=3,
                per_node_in=3,
                keep_all_depot_touches=True,
                depot_extra_topk=30,
            )
            print(f"{'='*60}")
            print("SAFE PRUNING COMPLETE")
            print(f"{'='*60}")
            print(f"Processed {len(results)} instances")
            print("Output saved to: safe_pruned_graphs/")
            print(f"{'='*60}")
            print(f"To solve with Gurobi, load:")
            print(f"  <instance_name>_edges_solver.csv")
            print(f"and keep <instance_name>_edges_scored_full.csv for analysis/debugging.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
