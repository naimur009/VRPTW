"""
predict.py  —  Inference script for EdgeRankGNNRefined
========================================================
Loads  models/best_edge_ranker_refined.pt  and runs edge-ranking
inference on new instances that have already been preprocessed into
the graph_data.pt format (same as produced by the training pipeline).

Usage
-----
# Run on default processed_data/ folder → saves to ranked_output/predict/
python predict.py

# Custom paths
python predict.py \\
    --data_root /path/to/new/processed_data \\
    --model_path models/best_edge_ranker_refined.pt \\
    --output_dir ranked_output/predict \\
    --device cpu

Output per instance
-------------------
  ranked_output/predict/<inst_name>.csv
  Columns: from, to, score, label, rank
    - from / to  : node IDs
    - score      : model confidence (0–1), higher = more likely in optimal route
    - label      : ground-truth edge label (1=positive, 0=negative).
                   Will be -1 if the graph_data.pt was built without labels.
    - rank       : 1 = highest-scored edge
"""

import os
import glob
import argparse
import math
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data

# ── Import architecture from model.py ─────────────────────────────────────────
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(os.path.dirname(script_dir), "3.training_model")
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from model import EdgeRankGNNRefined, recall_at_k, precision_at_k, ndcg_at_k


# ============================================================
# Dataset loader (lightweight — no split logic needed)
# ============================================================
class NewInstanceDataset:
    """
    Scans <root_dir>/**/graph_data.pt (recursive) and loads each one.
    Compatible with the graph_data.pt files produced by the training pipeline.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = sorted(
            glob.glob(os.path.join(root_dir, "**", "graph_data.pt"), recursive=True)
        )
        # Also accept flat layout: root_dir/graph_data.pt
        flat = os.path.join(root_dir, "graph_data.pt")
        if os.path.isfile(flat) and flat not in self.files:
            self.files.insert(0, flat)

        if not self.files:
            raise FileNotFoundError(
                f"No graph_data.pt files found under: {root_dir}\n"
                "Make sure your new instances have been pre-processed "
                "into graph_data.pt format (same as the training pipeline)."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        obj = torch.load(self.files[idx], map_location="cpu")

        # Build the Data object — labels are optional for pure prediction
        y = obj.get("y", None)
        if y is None:
            # No ground-truth labels → fill with -1 sentinel
            num_edges = obj["edge_index"].size(1)
            y = torch.full((num_edges,), -1.0)

        data = Data(
            x=obj["x"].float(),
            edge_index=obj["edge_index"].long(),
            edge_attr=obj["edge_attr"].float(),
            y=y.float(),
        )
        data.node_ids = obj["node_ids"].long()
        data.node_feature_cols = obj.get("node_feature_cols", [])
        data.edge_feature_cols  = obj.get("edge_feature_cols", [])
        data.instance_name = os.path.basename(os.path.dirname(self.files[idx]))
        data.source_file   = self.files[idx]
        return data


# ============================================================
# Model loader
# ============================================================
def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load EdgeRankGNNRefined from a .pt checkpoint."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    required_keys = {"model_state_dict", "node_dim", "edge_dim", "edge_feature_cols", "args"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        raise KeyError(
            f"Checkpoint is missing keys: {missing}\n"
            "Make sure you are using a checkpoint saved by model.py."
        )

    args = ckpt["args"]
    model = EdgeRankGNNRefined(
        node_dim=ckpt["node_dim"],
        edge_dim=ckpt["edge_dim"],
        edge_feature_cols=ckpt["edge_feature_cols"],
        hidden_dim=args.get("hidden_dim", 160),
        num_layers=args.get("num_layers", 5),
        dropout=args.get("dropout", 0.12),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Checkpoint  : {model_path}")
    print(f"  Best epoch  : {ckpt.get('best_epoch', '?')}")
    print(f"  Best score  : {ckpt.get('best_score', '?'):.4f}" if ckpt.get('best_score') else "")
    print(f"  Node dim    : {ckpt['node_dim']}")
    print(f"  Edge dim    : {ckpt['edge_dim']}")
    print(f"  Hidden dim  : {args.get('hidden_dim', 160)}")
    print(f"  Layers      : {args.get('num_layers', 5)}")
    return model


# ============================================================
# Per-instance inference
# ============================================================
@torch.no_grad()
def predict_instance(
    model: nn.Module,
    data: Data,
    device: torch.device,
    output_dir: str,
) -> str:
    """
    Run model inference on a single graph and save a ranked CSV.

    Returns the path of the saved CSV.
    """
    model.eval()
    data = data.to(device)

    logits = model(data.x, data.edge_index, data.edge_attr)
    scores = torch.sigmoid(logits).detach().cpu().numpy()

    src_idx  = data.edge_index[0].detach().cpu()
    dst_idx  = data.edge_index[1].detach().cpu()
    node_ids = data.node_ids.detach().cpu()

    from_nodes = node_ids[src_idx].numpy()
    to_nodes   = node_ids[dst_idx].numpy()
    labels     = data.y.detach().cpu().numpy()

    df = pd.DataFrame({
        "from" : from_nodes,
        "to"   : to_nodes,
        "score": scores,
        "label": labels,          # -1 if no ground truth was provided
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{data.instance_name}.csv")
    df.to_csv(out_file, index=False)
    return out_file


# ============================================================
# Optional per-instance metrics (only when labels are available)
# ============================================================
@torch.no_grad()
def compute_metrics(scores_np: np.ndarray, labels_np: np.ndarray) -> Optional[dict]:
    """Return recall / precision / NDCG metrics. Returns None if no labels."""
    if (labels_np < 0).all():          # sentinel -1 means no ground truth
        return None
    mask = labels_np >= 0
    s = torch.tensor(scores_np[mask], dtype=torch.float32)
    l = torch.tensor(labels_np[mask], dtype=torch.float32)
    return {
        "r@10%" : recall_at_k(s, l, 0.10),
        "r@15%" : recall_at_k(s, l, 0.15),
        "r@20%" : recall_at_k(s, l, 0.20),
        "p@10%" : precision_at_k(s, l, 0.10),
        "ndcg@10%": ndcg_at_k(s, l, 0.10),
    }


# ============================================================
# Batch inference
# ============================================================
def run_batch_inference(
    model: nn.Module,
    dataset: NewInstanceDataset,
    device: torch.device,
    output_dir: str,
) -> pd.DataFrame:
    """Iterate over all instances, predict, save, and collect a summary."""

    summary_rows = []
    total = len(dataset)
    width = len(str(total))

    print(f"\n{'─'*60}")
    print(f"  Predicting {total} instance(s) → {output_dir}")
    print(f"{'─'*60}")

    for idx in range(total):
        data = dataset[idx]
        out_file = predict_instance(model, data, device, output_dir)

        # Reload to compute metrics (avoids keeping everything in GPU memory)
        df = pd.read_csv(out_file)
        scores_np = df["score"].values
        labels_np = df["label"].values
        metrics = compute_metrics(scores_np, labels_np)

        row = {
            "instance"   : data.instance_name,
            "num_edges"  : len(df),
            "output_file": out_file,
        }
        if metrics:
            row.update(metrics)

        summary_rows.append(row)

        metric_str = ""
        if metrics:
            metric_str = (
                f"  R@10%={metrics['r@10%']:.3f}"
                f"  R@20%={metrics['r@20%']:.3f}"
                f"  NDCG@10%={metrics['ndcg@10%']:.3f}"
            )

        print(
            f"  [{idx+1:{width}d}/{total}]  {data.instance_name:<20s}"
            f"  edges={len(df):<6d}{metric_str}"
        )

    print(f"{'─'*60}")
    return pd.DataFrame(summary_rows)


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with best_edge_ranker_refined.pt on new VRPTW instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(script_dir, "Prepared_data"),
        help="Root directory containing inst_*/graph_data.pt files for new instances",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(script_dir, "models", "best_edge_ranker_refined.pt"),
        help="Path to the trained .pt checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(script_dir, "Ranked_output"),
        help="Directory where ranked CSVs will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detected if omitted.",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  VRPTW EDGE RANKER — INFERENCE MODE")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Output dir  : {args.output_dir}")
    print()

    # ── Load model ────────────────────────────────────────────
    print("Loading model …")
    model = load_model(args.model_path, device)

    # ── Load new instances ────────────────────────────────────
    print(f"\nScanning for graph_data.pt files in: {args.data_root}")
    dataset = NewInstanceDataset(args.data_root)
    print(f"  Found {len(dataset)} instance(s)")

    # ── Run inference ─────────────────────────────────────────
    summary_df = run_batch_inference(model, dataset, device, args.output_dir)

    # ── Save summary ──────────────────────────────────────────
    summary_csv = os.path.join(args.output_dir, "prediction_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary saved → {summary_csv}")

    # ── Print aggregate metrics if labels were present ────────
    metric_cols = [c for c in summary_df.columns if c.startswith("r@") or c.startswith("p@") or c.startswith("ndcg")]
    if metric_cols:
        print("\n  Aggregate metrics (mean across instances):")
        for col in metric_cols:
            print(f"    {col:<12s}: {summary_df[col].mean():.4f}")

    print(f"\n  Done. {len(dataset)} ranked CSV(s) saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
