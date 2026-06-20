# VRPTW Research Project: Multi-Phase Optimization with GNN Edge Ranking

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Directory Structure](#3-directory-structure)
4. [Architecture](#4-architecture)
5. [Technologies & Dependencies](#5-technologies--dependencies)
6. [Modules in Detail](#6-modules-in-detail)
7. [Data Pipeline](#7-data-pipeline)
8. [Algorithms & Notable Implementations](#8-algorithms--notable-implementations)
9. [Setup & Execution](#9-setup--execution)
10. [Configuration](#10-configuration)

---

## 1. Project Overview

This project solves the **Vehicle Routing Problem with Time Windows (VRPTW)** using a hybrid machine learning pipeline. The core idea is to train a **Graph Neural Network (GNN)** to predict which edges (customer-to-customer or depot-to-customer arcs) are likely part of optimal routes, then use those predictions to prune the search space for downstream solvers — making them faster and more effective.

The pipeline spans four phases: synthetic data generation, ground-truth solving via genetic algorithms, GNN training for edge ranking, and evaluation on pruned graphs with an exact MILP solver.

---

## 2. Objectives

- **Generate** realistic Solomon-style synthetic VRPTW benchmark instances with controlled difficulty (via a Dynamic Edge Selection / DER methodology).
- **Obtain ground-truth optimal routes** using a genetic algorithm (PyVRP) with a two-phase optimization strategy.
- **Train a GNN** (`EdgeRankGNNRefined`) to rank edges by their likelihood of belonging to an optimal solution, using 9 node features and 36 edge features.
- **Prune the search space** of new VRPTW instances by retaining only top-ranked edges, then solve the reduced problem with Gurobi.
- **Evaluate** the trade-off between graph sparsity and solution quality, comparing GNN-pruned solving against a pure Gurobi baseline.

---

## 3. Directory Structure

```
VRPTW/
├── .gitignore
├── .vscode/
│   └── settings.json
├── README.md
├── requirements.txt
├── project.md                          # This file
│
├── data/
│   └── Raw-Data/
│       └── 100_customer/
│           ├── C1/ C2/                 # Clustered customers, narrow/wide windows
│           ├── R1/ R2/                 # Random customers, narrow/wide windows
│           └── RC1/ RC2/               # Mixed clustered-random, narrow/wide windows
│
├── Gurobi_Solver/
│   └── solver.py                       # Pure Gurobi baseline (full graph, no pruning)
│
└── research_ready/
    ├── 1.Data_genaration/
    │   ├── Data_generation_using_DER.py # Synthetic Solomon-style instance generator
    │   └── data/                        # Generated CSV instances (per series)
    │       ├── C1/ C2/ R1/ R2/ RC1/ RC2/
    │
    ├── 2.Genetic_Algorithm/
    │   ├── pyvrp.py                     # Batch GA solver with Rich dashboard
    │   └── dataset/                     # Solved instances (data.csv + solution.txt)
    │       ├── batch_summary.csv
    │       ├── C1/ C2/ R1/ R2/ RC1/ RC2/
    │
    ├── 3.training_model/
    │   ├── model.py                     # GNN edge ranking model (training loop)
    │   ├── dataCleaning.py              # Preprocess instances → PyTorch Geometric graphs
    │   ├── dataset/                     # Raw CSVs from GA solver
    │   ├── models/                      # Saved model checkpoints
    │   ├── processed_data/              # 1500 inst_* folders, each with:
    │   │   └── inst_*_D*_****/
    │   │       ├── edge_features.csv
    │   │       ├── edge_index.csv
    │   │       ├── graph_data.pt
    │   │       ├── node_features.csv
    │   │       ├── stats.json
    │   │       └── y_label.csv
    │   ├── ranked_output/               # GNN-ranked edge CSVs
    │   └── splits/
    │       ├── train.txt                # 1200 training instances
    │       └── val.txt                  # 300 validation instances
    │
    └── 4.Evaluation_testing_result/
        ├── predict.py                   # GNN inference on new instances
        ├── preprocess_test.py           # Lightweight preprocessing for test data
        ├── build_solver_graph.py        # Build pruned graphs from ranked edges
        └── solver.py                    # Gurobi on GNN-pruned graphs (dynamic subsets)
```

---

## 4. Architecture

The project follows a **multi-phase pipeline architecture** with four main stages:

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA GENERATION                                        │
│  Data_generation_using_DER.py  ──> Solomon-style CSV instances   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2: GROUND TRUTH SOLVING (Genetic Algorithm)               │
│  pyvrp.py  ──> solution.txt + data.csv                          │
│  Two-phase: minimize vehicles, then minimize distance            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3: GNN TRAINING                                           │
│  dataCleaning.py  ──> graph_data.pt (node/edge features + labels)│
│  model.py  ──> EdgeRankGNNRefined model                         │
│            ──> ranked_output/*.csv (edge scores)                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 4: INFERENCE & EVALUATION                                 │
│  predict.py  ──> rank edges on new instances                     │
│  build_solver_graph.py  ──> pruned subgraphs (top-k edges)       │
│  solver.py  ──> Gurobi on pruned graphs (dynamic subset sizes)   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
CSV Instances  ──>  GA Solver  ──>  Solution (ground truth)
     │                                    │
     └────>  dataCleaning.py  <────────────┘
                      │
                      ▼
            graph_data.pt (features + labels)
                      │
                      ▼
              model.py (train GNN)
                      │
                      ▼
            trained model checkpoint
                      │
                      ▼
    predict.py ──> ranked edges ──> build_solver_graph.py ──> pruned graphs ──> solver.py (Gurobi)
```

---

## 5. Technologies & Dependencies

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Primary language |
| **PyTorch** | 2.12.1 | Deep learning framework |
| **PyTorch Geometric** | 2.8.0 | GNN operations (GINEConv, GraphNorm) |
| **Gurobi** | 13.0.1 | Exact MILP solver for VRPTW |
| **PyVRP** | 0.13.3 | Genetic algorithm VRP solver |
| **NumPy** | 2.4.6 | Numerical computation |
| **Pandas** | 3.0.3 | Data manipulation |
| **SciPy** | 1.17.1 | Statistical distributions |
| **Rich** | 15.0.0 | Terminal dashboards & UI |

Full list in `requirements.txt`.

---

## 6. Modules in Detail

### Phase 1: Data Generation (`1.Data_genaration/Data_generation_using_DER.py`)

Generates synthetic Solomon-style benchmark instances using a **DER (Dynamic Edge Selection)** methodology.

- **Coordinate generation:** Clustered (C-series via seed points + Gaussian noise), random uniform (R-series), or mixed (RC-series).
- **Demand sampling:** Series-specific distributions with scaling to fit 75–95% fleet capacity.
- **Time window sampling:** Fitted statistical distributions (beta, gamma, GEV, Weibull) for half-widths. Density parameter (`d25/d50/d75/d100`) controls the fraction of customers with tight windows.
- **Output format:** CSVs with columns `CUST_NO, XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE, SERVICE TIME`.
- **Validation:** Basic feasibility checks + optional PyVRP solver validation.

### Phase 2: Genetic Algorithm Solver (`2.Genetic_Algorithm/pyvrp.py`)

Batch VRPTW solver using PyVRP with a real-time **Rich terminal dashboard**.

- **Two-phase strategy:**
  1. **Minimize vehicles:** Iteratively reduces fleet size until infeasible.
  2. **Minimize distance:** Optimizes total travel distance with the minimum feasible fleet.
- **Dashboard features:** Instance progress bars, event logs, results table, ETA.
- **Output:** `solution.txt` (route sequences) and `data.csv` per instance.

### Phase 3a: Data Preprocessing (`3.training_model/dataCleaning.py`)

Converts raw CSV instances + solution files into PyTorch Geometric graph objects.

- **Node features (9):** `x_norm, y_norm, demand_norm, ready_time_norm, due_date_norm, service_time_norm, window_width_norm, depot_dist_norm, is_depot`.
- **Edge features (36):** Geometric (distance, dx/dy), node context on both endpoints, time-window heuristics (arrival time, waiting time, slack, overlap, feasibility), depot context.
- **Labeling:** Edges appearing in the GA solution routes are positive (1); all others are negative (0).
- **Output:** `graph_data.pt` containing `x, edge_index, edge_attr, y, node_ids`.

### Phase 3b: GNN Model (`3.training_model/model.py`)

Defines and trains the **EdgeRankGNNRefined** architecture.

**Architecture:**
```
EdgeRankGNNRefined:
  - node_encoder: MLP (input_dim ──> 160)
  - edge_encoder: MLP (edge_dim + 1 ──> 160)
  - prior_net: MLP (edge_dim ──> 32 ──> 1) [learnable prior]
  - 5× GINEConv layers with:
      * GINEConv message passing
      * GraphNorm + GELU activation
      * ResidualMLP (FFN + LayerNorm)
      * EdgeUpdate (src/dst/edge concat → MLP → residual)
  - Edge head: 5-layer MLP (hidden_dim*5 ──> 160 ──> 80 ──> 1)
      Input: [x_src, x_dst, pair_mul, pair_diff, e]
```

**Loss function (triple objective):**
1. **Focal BCE** with label smoothing and positive-class weighting.
2. **Pairwise ranking loss** with hardness-weighted negative sampling.
3. **Top-k recall surrogate** — differentiable approximation of Recall@k.

**Training details:** AdamW optimizer, linear warmup + cosine LR schedule, gradient clipping, AMP (mixed precision), optional `torch.compile`.

**Metrics:** Recall@k (10%/15%/20%), Precision@10%, NDCG@10%, Average Precision, Avg Positive Rank Percentile.

**Validation score:** Weighted combination: 30% AP + 25% R@10 + 20% R@15 + 10% P@10 + 10% NDCG@10 + 5% rank percentile.

**Data split:** Family-level (R1/C1/RC1 → train; R2/C2/RC2 → val) to prevent information leakage.

### Phase 4a: Inference (`4.Evaluation_testing_result/predict.py`)

Runs trained model on new instances:
- Loads `graph_data.pt` recursively.
- Produces ranked edge CSV: `from, to, score, label, rank`.
- Computes recall/precision/NDCG if labels are available.
- Outputs `prediction_summary.csv`.

### Phase 4b: Build Solver Graph (`4.Evaluation_testing_result/build_solver_graph.py`)

Converts ranked edges into pruned subgraphs for Gurobi:
1. Takes **top-k%** edges by GNN score.
2. Ensures all depot-to-customer and customer-to-depot edges are included.
3. Adds **support edges**: top-k outgoing and top-k incoming per customer.
4. Repairs low-degree nodes (minimum in/out degree guarantee).
5. Outputs `solver_edges_top_{PCT}.csv` and `nodes.csv`.

### Phase 4c: Pruned-Graph Solver (`4.Evaluation_testing_result/solver.py`)

Dynamic Gurobi solver that tries progressively larger edge subsets:
- **Subset strategy:** 15% → 20% → 25% → 30% → 40% → 50% (stops at first feasible solution).
- **Constraints:** Flow conservation, time window propagation (Big-M), capacity/load propagation (MTZ-style).
- **Features:** Quick pre-check (connectivity + capacity), Gurobi callback for progress, Rich dashboard.
- **Output:** `dynamic_result.json` per instance + `summary.csv`.

### Baseline Solver (`Gurobi_Solver/solver.py`)

Pure Gurobi MILP on the full graph (no pruning) for comparison. Weighted objective `λ · num_vehicles + total_distance`. Includes IIS computation for infeasible models.

---

## 7. Data Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Generate     │    │  GA Solve    │    │  Preprocess  │    │  Train GNN   │
│  Instances    │───▶│  (Ground     │───▶│  → graph     │───▶│  (EdgeRank   │
│  (DER)        │    │   Truth)     │    │   data.pt    │    │   GNNRefined)│
└──────────────┘    └──────────────┘    └──────────────┘    └───────┬──────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Evaluate    │    │  Solve with  │    │  Build       │    │  Inference   │
│  (compare    │◀───│  Pruned      │◀───│  Solver      │◀───│  (predict.py)│
│   results)   │    │  Graphs      │    │  Graphs      │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Data volume:** 1500 instances total — 1200 train / 300 validation, spanning 6 Solomon series families (C1, C2, R1, R2, RC1, RC2) at 4 density levels (d25, d50, d75, d100).

---

## 8. Algorithms & Notable Implementations

### GNN: EdgeRankGNNRefined
- **GINEConv** layers for edge-aware message passing.
- **GraphNorm** for training stability.
- **ResidualMLP** with pre-activation residual blocks and LayerNorm.
- **EdgeUpdate**: updates edge features by concatenating source, destination, and current edge representations through an MLP with residual connection.
- **Gated interaction**: `sigmoid(src_gate + dst_gate + edge_gate) · (src · dst)` for pairwise edge scoring.
- **Dual prior**: hand-crafted heuristic prior + learned prior from edge attributes.

### Loss Functions
- **Focal BCE**: down-weights easy negatives, focuses on hard misclassifications.
- **Pairwise ranking loss**: samples positive-negative pairs with hardness-weighted negative sampling (softmax + multinomial).
- **Top-k recall surrogate**: differentiable approximation using the k-th score as a threshold.

### Data Generation (DER)
- Statistical distribution fitting (beta, gamma, GEV, Weibull) on real Solomon instance time window half-widths.
- Cluster seeds for C-series with Gaussian noise perturbation.
- Demand scaling to 75–95% fleet capacity utilization.
- Dual validation: basic feasibility + solver-based verification.

### Two-Phase GA Strategy
- Phase 1: Binary search / iterative reduction of vehicle count until infeasibility.
- Phase 2: Distance minimization with fixed fleet size.

### Dynamic Subset Solving
- Iterates through increasing edge subset percentages.
- Stops at first feasible solution (greedy for speed).
- Gurobi callback for real-time MIP gap monitoring.

---

## 9. Setup & Execution

### Prerequisites
- Python 3.10+
- Gurobi license (free academic at gurobi.com)
- NVIDIA GPU recommended (CPU works but slower)

### Installation

```bash
# Virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch (GPU)
pip install torch==2.12.1 torch-geometric==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
pip install -r requirements.txt
```

### Pipeline Execution

| Step | Command | Description |
|---|---|---|
| 1 | `python research_ready/1.Data_genaration/Data_generation_using_DER.py --n_instances 100 --series R1 R2` | Generate synthetic instances |
| 2 | `python research_ready/2.Genetic_Algorithm/pyvrp.py --phase1-time 10 --phase2-time 30` | Solve for ground truth |
| 3 | `python research_ready/3.training_model/dataCleaning.py --input_root path/to/dataset --output_root path/to/processed_data --save_pt` | Preprocess for GNN |
| 4 | `python research_ready/3.training_model/model.py --data_root path/to/data --epochs 80` | Train GNN |
| 5 | `python research_ready/4.Evaluation_testing_result/predict.py --model_path models/best_edge_ranker_refined.pt` | Run inference |
| 6 | `python research_ready/4.Evaluation_testing_result/build_solver_graph.py` | Build pruned graphs |
| 7 | `python research_ready/4.Evaluation_testing_result/solver.py --time_limit 5 --subsets 15 20 25 30 40 50` | Solve with pruning |
| Baseline | `python Gurobi_Solver/solver.py` | Pure Gurobi baseline |

---

## 10. Configuration

Most configuration is inline at the top of each script:

| Script | Configurable Parameters |
|---|---|
| `Data_generation_using_DER.py` | Series, n_customers, density levels, n_instances per config |
| `pyvrp.py` | Phase 1/2 time limits, seed, population size |
| `model.py` | Learning rate, batch size, epochs, hidden dimensions, loss weights, early stopping patience |
| `predict.py` | Model path, batch size, device |
| `build_solver_graph.py` | Top-k percentages, support edge counts |
| `solver.py` | Time limit, MIP gap, subset percentages |

Split files for train/validation: `splits/train.txt` (1200 instances) and `splits/val.txt` (300 instances), using family-level splitting to prevent leakage.
