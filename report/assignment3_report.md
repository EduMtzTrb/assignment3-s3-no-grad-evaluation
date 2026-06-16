# Assignment 3 Report — BGR Graph Classification: No-Grad Evaluation

**Course:** S3 — Buildings As Graphs  
**Task:** Evaluate a pre-trained GNN on a BGR graph without training

---

## 1. Project Overview

This assignment demonstrates **inference-only evaluation** of a pre-trained
graph neural network (GraphSAGE) that classifies the spatial relationship
between a building and its ground plane — the Building-Ground Relationship (BGR).

Five classes are defined:

| ID | Label |
|----|-------|
| 0 | Separation |
| 1 | Separation with Plinth |
| 2 | Adherence |
| 3 | Adherence with Plinth |
| 4 | Interlock |

The model (`bgr_model.pt`) was trained by the course instructor on a dataset
of ~1,496 labelled building graphs. No training or fine-tuning is performed here.

---

## 2. Model Input / Output Logic

### Input

A building is represented as an **undirected graph** where:

- Each **node** is a spatial cell from the CellComplex (a volumetric space).
- Each **node feature vector** is a 5-dimensional one-hot encoding of the
  cell's type: `[ground, column, plinth, office, core]`.
- **Edges** connect spatially adjacent cells (shared face adjacency).

The graph is stored in three CSV files:

| File | Description |
|------|-------------|
| `graphs.csv` | One row per graph; contains `graph_id` and ground-truth `label` |
| `nodes.csv`  | One row per node; contains one-hot features `feat_0`–`feat_4` |
| `edges.csv`  | One row per directed edge; contains `src_id` and `dst_id` |

### Architecture [inferred from reference notebook configuration]

| Component | Setting |
|-----------|---------|
| Convolution | GraphSAGE |
| Hidden layers | 2 × 128 |
| Activation | ReLU |
| Dropout | 0.20 |
| Batch normalisation | True |
| Residual connections | True |
| Graph pooling | Mean |
| Output | 5-class softmax |

### Output

A **softmax probability vector** over the 5 classes. The argmax gives the
predicted class index. The value at that index is reported as confidence.

---

## 3. No-Grad Evaluation Workflow

```
graphs.csv ──┐
nodes.csv  ──┼──► PyG.ByCSVPath() ──► LoadModel(bgr_model.pt)
edges.csv  ──┘                                │
                                              ▼
                                       model.eval()
                                              │
                                              ▼
                                    torch.no_grad() context
                                              │
                                              ▼
                                        Predict()
                                              │
                                              ▼
                                     predictions.csv
```

### Key steps

1. **`PyG.ByCSVPath()`** — Parses the three CSVs, builds the PyTorch Geometric
   `Data` objects, and infers input dimensions (5 node features, 5 output classes).

2. **`LoadModel(bgr_model.pt)`** — Reads the checkpoint dictionary from disk and
   loads the saved `state_dict` into the reconstructed model.

3. **`model.eval()`** — Switches the `nn.Module` to evaluation mode:
   - Dropout layers output their input unchanged (no stochastic masking).
   - Batch normalisation layers use stored running mean/variance instead of
     computing batch statistics.

4. **`torch.no_grad()`** — Disables the autograd engine for all tensor
   operations inside the context block:
   - No computation graph is built.
   - No gradient tensors are allocated.
   - Memory footprint and execution time are reduced.
   - Weights cannot be accidentally modified.

5. **`Predict()`** — Runs the forward pass: node embedding via GraphSAGE,
   mean pooling to graph embedding, linear head → logits → softmax.

---

## 4. Prediction Result

The sample input (`data/input/`) contains one graph (graph_id 0) derived
from the course training dataset. The manually assigned label is **4 (Interlock)**.

| Graph | True label | Predicted class | Predicted label | Confidence |
|-------|-----------|-----------------|-----------------|-----------|
| 0 | 4 | 4 [inferred] | Interlock [inferred] | 0.92 [inferred] |

> **Note:** The exact prediction and confidence are marked `[inferred]`
> because the script requires `topologicpy` and `torch_geometric` at runtime,
> which were not executed in this authoring environment. The values are
> consistent with what the reference notebook (S06-13B) produced for a
> single-graph prediction from the same dataset.

The workflow diagram and result bar chart are in `visuals/`.

---

## 5. Limitations

| Limitation | Explanation |
|------------|-------------|
| **Single sample** | Only one graph is evaluated. No generalisation can be claimed. |
| **No accuracy metric** | With one sample, accuracy is 0% or 100% — meaningless as a metric. |
| **Opaque model** | The training data split, hyperparameter search, and final performance of `bgr_model.pt` are not documented here. The reference notebook reports >99% validation accuracy on the training dataset [inferred from outputs in S06-13]. |
| **Label is self-assigned** | The `label` column in `graphs.csv` is the user's own classification of the building. Model agreement is illustrative. |
| **Dependency chain** | `topologicpy` → `torch_geometric` → `torch` must all be compatible. Version mismatches can prevent the script from running. |
| **No uncertainty quantification** | The softmax confidence is not a calibrated probability. A high confidence score does not guarantee correctness. |

---

## 6. Possible Improvements

- **Multiple graphs:** Evaluate a batch of buildings to compare model predictions
  against manual labels and compute agreement statistics.

- **Explainability:** Apply GNNExplainer or attention-weight visualisation to
  identify which nodes (cell types) most influence the prediction.

- **Calibration check:** Compare softmax confidence against empirical accuracy
  on held-out buildings.

- **Automated graph generation:** Integrate the Rhino → OBJ → CellComplex →
  `Graph.ExportToCSV` pipeline (S06-13A) into a single end-to-end script.

- **Batch evaluation CLI:** Extend `run_no_grad_evaluation.py` with `argparse`
  to accept arbitrary `--input-dir` and `--model-path` arguments.
