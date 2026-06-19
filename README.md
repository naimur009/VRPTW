# VRPTW Research Project: Multi-Phase Optimization with GNN Edge Ranking

A comprehensive framework for solving the **Vehicle Routing Problem with Time Windows (VRPTW)**. This project integrates statistical data generation, Graph Neural Networks (GNN) for edge prioritization, and hybrid solvers (Genetic Algorithms & Exact Solvers) to achieve high-quality routing solutions.

---

## 📂 Repository Structure

The project is organized into modular phases representing the research pipeline:

### `research_ready/`
*   **`1.Data_genaration/`**: Script `Data_generation_using_DER.py` for generating synthetic Solomon-like instances using Dynamic Edge Selection (DER) methodology.
*   **`2.Genetic_Algorithm/`**: `pyvrp.py` dashboard-driven batch solver utilizing the [PyVRP](https://github.com/pyvrp/pyvrp) library.
*   **`3.training_model/`**: GNN implementation (`model.py`) for edge ranking. Uses GINEConv layers to predict the probability of an edge being part of the optimal route.
*   **`4.Evaluation_testing_result/`**: Inference scripts (`predict.py`) and evaluation solvers to validate model performance on unseen instances.

### `Gurobi_Solver/`
*   Contains `solver.py`, a Gurobi-based exact solver used to generate ground-truth optimal solutions for training and benchmarking.

---

## 🚀 Research Pipeline

### 1. Instance Generation
Generate diverse VRPTW instances with varying customer densities:
```bash
python research_ready/1.Data_genaration/Data_generation_using_DER.py --n_instances 100 --series R1 R2
```

### 2. GNN Training
Train the edge-ranking model using generated graph data:
```bash
python research_ready/3.training_model/model.py --data_root path/to/data --epochs 80
```

### 3. Inference & Prediction
Rank edges for new instances using a trained model:
```bash
python research_ready/4.Evaluation_testing_result/predict.py --model_path models/best_edge_ranker_refined.pt
```

### 4. Guided Optimization
Solve VRP instances using the guided Genetic Algorithm or Exact Solver:
```bash
python research_ready/2.Genetic_Algorithm/pyvrp.py --phase1-time 10 --phase2-time 30
```

---

## ⚙️ Installation

1. **Prerequisites**: Python 3.10+, Gurobi license (free academic at gurobi.com).

2. **Setup**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **No NVIDIA GPU?** PyPI ships the CUDA version of torch (~530MB). If torch fails to import, reinstall the CPU version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

---

## 💻 Environment & Hardware Note
*   **Data Generation & Solvers**: Primarily CPU-bound.
*   **GNN Training/Inference**: Supports both CPU and GPU. For large-scale training, an NVIDIA GPU is highly recommended for performance.
*   **Current Setup Check**:
    ```python
    import torch
    print(torch.cuda.is_available())  # Checks for GPU detection
    ```

---

## 📊 Summary of Key Technologies
*   **Logic**: Python, PyVRP, Gurobi.
*   **Deep Learning**: PyTorch, PyTorch Geometric.
*   **Numerics**: NumPy, SciPy, Pandas.
*   **UI**: Rich (for terminal dashboards).