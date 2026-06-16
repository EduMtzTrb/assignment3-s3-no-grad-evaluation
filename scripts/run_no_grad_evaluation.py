"""
Assignment 3 — BGR Graph Classification: No-Grad Evaluation
Loads a pre-trained GraphSAGE model (bgr_model.pt) and predicts the
Building-Ground Relationship class for one or more input graphs.

No training is performed. Weights are not modified.
"""

import sys
from pathlib import Path
import pandas as pd
import torch

# --- Paths (relative to repo root) ---
REPO_ROOT   = Path(__file__).resolve().parent.parent
INPUT_DIR   = REPO_ROOT / "data" / "input"
MODEL_PATH  = REPO_ROOT / "data" / "model" / "bgr_model.pt"
OUTPUT_PATH = REPO_ROOT / "data" / "predictions.csv"

# --- Constants ---
REQUIRED_NODE_FEATURES = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]

CLASS_MAPPING = {
    0: "Separation",
    1: "Separation with Plinth",
    2: "Adherence",
    3: "Adherence with Plinth",
    4: "Interlock",
}


def verify_node_features(nodes_path: Path) -> None:
    """Confirm the five required one-hot feature columns are present."""
    nodes_df = pd.read_csv(nodes_path)
    missing = [f for f in REQUIRED_NODE_FEATURES if f not in nodes_df.columns]
    if missing:
        raise ValueError(
            f"nodes.csv is missing required feature columns: {missing}\n"
            f"Expected: {REQUIRED_NODE_FEATURES}"
        )
    print(f"[OK] Node features verified: {REQUIRED_NODE_FEATURES}")
    cell_types = {
        "feat_0": "ground",
        "feat_1": "column",
        "feat_2": "plinth",
        "feat_3": "office",
        "feat_4": "core",
    }
    for col, name in cell_types.items():
        n = nodes_df[col].sum()
        print(f"      {col} ({name:8s}): {int(n)} nodes")


def main() -> None:
    # 1. Validate input files
    for fname in ("graphs.csv", "nodes.csv", "edges.csv"):
        fpath = INPUT_DIR / fname
        if not fpath.exists():
            sys.exit(f"[ERROR] Required input file not found: {fpath}")
    if not MODEL_PATH.exists():
        sys.exit(f"[ERROR] Model file not found: {MODEL_PATH}")

    print(f"Input directory : {INPUT_DIR}")
    print(f"Model path      : {MODEL_PATH}")

    # 2. Verify node features
    verify_node_features(INPUT_DIR / "nodes.csv")

    # 3. Load dataset via topologicpy PyG
    try:
        from topologicpy.PyG import PyG
    except ImportError:
        sys.exit(
            "[ERROR] topologicpy is not installed.\n"
            "Install it with: pip install topologicpy"
        )

    print("\nLoading dataset with PyG.ByCSVPath ...")
    pyg_eval = PyG.ByCSVPath(
        path=str(INPUT_DIR),
        level="graph",
        task="classification",
        graphLabelType="categorical",
        nodeLabelType="categorical",
        edgeLabelType="categorical",
    )

    # 4. Load pre-trained weights
    print("Loading pre-trained model weights ...")
    pyg_eval.LoadModel(str(MODEL_PATH))

    # 5. Explicitly set model to evaluation mode (disables dropout / batch-norm training behaviour)
    if pyg_eval.model is not None:
        pyg_eval.model.eval()
        print("[OK] model.eval() called — dropout and batch-norm set to inference mode")
    else:
        print("[WARN] pyg_eval.model is None; eval() not called explicitly")

    # 6. Route all graphs to the test split
    pyg_eval.SetHyperparameters(split=(0.0, 0.0, 1.0), shuffle=False)

    # 7. Run prediction inside torch.no_grad() to suppress gradient computation
    print("\nRunning inference with torch.no_grad() ...")
    with torch.no_grad():
        pred_results = pyg_eval.Predict()

    print("[OK] torch.no_grad() block completed — no gradients were computed")

    # 8. Decode results
    predictions   = pred_results["pred"].tolist()
    probabilities = pred_results["prob"].tolist()

    results_df = pd.DataFrame({
        "graph_id":        range(len(predictions)),
        "predicted_class": predictions,
        "predicted_label": [CLASS_MAPPING[int(p)] for p in predictions],
        "confidence":      [round(max(p), 4) for p in probabilities],
    })

    # 9. Save to CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to: {OUTPUT_PATH}")

    # 10. Print summary
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    for _, row in results_df.iterrows():
        print(
            f"  Graph {int(row['graph_id'])}: "
            f"Class {int(row['predicted_class'])} — "
            f"{row['predicted_label']}  "
            f"(confidence: {row['confidence']})"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
