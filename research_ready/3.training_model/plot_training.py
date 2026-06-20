import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_training(csv_path: str = "models/training_log.csv", save_path: str = "training_curves.png") -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_path) if not os.path.isabs(csv_path) else csv_path

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VRPTW GNN Training Curves", fontsize=16, fontweight="bold", y=1.02)

    # 1. Learning Curve — Average Precision (main ranking metric)
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_ap"], label="Train AP", color="#2E86AB", linewidth=2)
    ax.plot(df["epoch"], df["val_ap"], label="Val AP", color="#A23B72", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Precision")
    ax.set_title("Learning Curve (AP)")
    ax.legend(loc="lower right")
    ax.set_xlim(1, df["epoch"].max())

    # 2. Train vs Validation Loss
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#2E86AB", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#A23B72", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend(loc="upper right")
    ax.set_xlim(1, df["epoch"].max())

    # 3. Constraint Violation Rate
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["train_violation_rate"], label="Train Violation Rate", color="#2E86AB", linewidth=2)
    ax.plot(df["epoch"], df["val_violation_rate"], label="Val Violation Rate", color="#A23B72", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Constraint Violation Rate (Time-Infeasible Edges)")
    ax.legend(loc="upper right")
    ax.set_xlim(1, df["epoch"].max())

    # 4. Prediction Entropy
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["train_entropy"], label="Train Entropy", color="#2E86AB", linewidth=2)
    ax.plot(df["epoch"], df["val_entropy"], label="Val Entropy", color="#A23B72", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.set_title("Prediction Entropy (Model Confidence)")
    ax.legend(loc="lower left")
    ax.set_xlim(1, df["epoch"].max())
    ax.axhline(y=0.693, color="gray", linestyle="--", alpha=0.5, label="Max uncertainty")
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(script_dir, save_path) if not os.path.isabs(save_path) else save_path
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved training curves to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot VRPTW GNN training curves from CSV log")
    parser.add_argument("--csv", type=str, default="models/training_log.csv", help="Path to training_log.csv")
    parser.add_argument("--output", type=str, default="training_curves.png", help="Output image path")
    args = parser.parse_args()
    plot_training(csv_path=args.csv, save_path=args.output)
