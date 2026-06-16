# Assignment 3 — BGR Graph Classification: No-Grad Evaluation

Graph-level classification of a building's relationship to the ground using a
pre-trained GraphSAGE model and `torch.no_grad()`.

---

## Project Structure

```
assignment3-s3-no-grad-evaluation/
├── data/
│   ├── input/          # Sample BGR graph (graphs.csv, nodes.csv, edges.csv)
│   ├── model/          # Pre-trained model (bgr_model.pt)
│   └── predictions.csv # Output written by the evaluation script
├── notebooks/
│   └── assignment3_no_grad_evaluation.ipynb
├── reference/
│   └── course_notebooks/   # S06-13B GML Predict BGR Graph.ipynb, etc.
├── report/
│   └── assignment3_report.md
├── scripts/
│   └── run_no_grad_evaluation.py
└── visuals/
    ├── evaluation_workflow.png
    ├── input_graph.png
    └── prediction_result.png
```

---

## Quick Start

### Requirements

```bash
pip install topologicpy torch pandas
# torch-geometric: follow https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### Run the evaluation

```bash
python scripts/run_no_grad_evaluation.py
```

Expected output:

```
Input directory : .../data/input
Model path      : .../data/model/bgr_model.pt
[OK] Node features verified: ['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4']
Loading dataset with PyG.ByCSVPath ...
Loading pre-trained model weights ...
[OK] model.eval() called
Running inference with torch.no_grad() ...
[OK] torch.no_grad() block completed — no gradients were computed

Predictions saved to: .../data/predictions.csv

============================================================
PREDICTION RESULTS
============================================================
  Graph 0: Class 4 — Interlock  (confidence: 0.92)
============================================================
```

---

## Input Format

Three CSV files must be present in `data/input/`:

| File | Key columns |
|------|------------|
| `graphs.csv` | `graph_id`, `label` |
| `nodes.csv`  | `graph_id`, `node_id`, `label`, `feat_0`…`feat_4` |
| `edges.csv`  | `graph_id`, `src_id`, `dst_id`, `label` |

Node features are a one-hot encoding of cell type:

| Column | Cell type |
|--------|----------|
| feat_0 | ground |
| feat_1 | column |
| feat_2 | plinth |
| feat_3 | office |
| feat_4 | core |

---

## Model

- **File:** `data/model/bgr_model.pt`
- **Architecture:** GraphSAGE with two hidden layers (128, 128), mean pooling
- **Task:** 5-class graph-level classification
- **Trained by:** Course instructor (weights not modified here)

---

## Output

`data/predictions.csv` — one row per input graph:

```
graph_id, predicted_class, predicted_label, confidence
0, 4, Interlock, 0.92
```

---

## Reference Notebooks

| Notebook | Purpose |
|----------|---------|
| S06-13A GML Creating BGR Graph | Build graph CSVs from Rhino geometry |
| S06-13B GML Predict BGR Graph  | Load model and predict BGR class |
| S06-13 GML Graph Classification | Train the GraphSAGE model |
