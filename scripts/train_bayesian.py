"""
Train Bayesian classifier (Gaussian Naive Bayes): load cached features, tune on val, evaluate on test.

Execution in plain English:
  1. Load the precomputed feature matrices (X) and labels (y) for train, val, and test.
  2. Call the Bayesian training routine: it scales the data, runs grid search using the
     validation set to pick the best var_smoothing (smoothing for class/feature stats),
     then refits the best model on the training set only.
  3. Scale the test set with the same scaler, predict with the trained model (each test
     point is assigned the class with highest posterior probability under the fitted
     Gaussian per-class distributions).
  4. Compute metrics (balanced accuracy, macro F1) and save them plus a confusion
     matrix plot to the results folder.

Usage: python scripts/train_bayesian.py [--data-config configs/data.yaml] [--model-config configs/model_bayesian.yaml]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

# Add project root to Python path so we can import from src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.data.dataset import ACTION_NAMES
from src.models.bayesian import train_bayesian, load_bayesian_config
from src.evaluation.metrics import classification_metrics, plot_confusion_matrix


def load_data_config(path: Path) -> dict:
    """Read the data config YAML (paths, n_frames, feature params, split ratios)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_cached_features(processed_dir: Path, data_cfg: dict):
    """
    Load the cached feature arrays and labels for train, val, and test.
    The cache key is built from the data config (e.g. n_frames, grid size, HOF bins)
    so we load the same feature version that was produced by run_preprocess.py.
    Returns a dict with keys 'train', 'val', 'test'; each value is (X, y).
    """
    cache_key = "n{n_frames}_b{n_blocks_h}x{n_blocks_w}_hof{hof_bins}_{feature_version}".format(
        n_frames=data_cfg["n_frames"],
        n_blocks_h=data_cfg.get("n_blocks_h", 5),
        n_blocks_w=data_cfg.get("n_blocks_w", 5),
        hof_bins=data_cfg.get("hof_bins", 9),
        feature_version=data_cfg.get("feature_version", "v1"),
    )
    out = {}
    for split in ["train", "val", "test"]:
        f = processed_dir / f"features_{split}_{cache_key}.npz"
        if not f.exists():
            raise FileNotFoundError(f"Run preprocess first: {f}")
        d = np.load(f)
        out[split] = (d["X"], d["y"])
    return out


def main():
    # --- Parse command-line options (config file paths and where to write results) ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=Path, default=ROOT / "configs" / "data.yaml")
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model_bayesian.yaml")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "bayesian")
    args = ap.parse_args()

    # --- Load configs and ensure results directory exists ---
    data_cfg = load_data_config(args.data_config)
    model_cfg = load_bayesian_config(args.model_config)
    processed_dir = ROOT / data_cfg["processed_dir"]
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load cached features: one feature matrix (X) and label vector (y) per split ---
    data = get_cached_features(processed_dir, data_cfg)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # --- Train Gaussian NB: scaling, grid search on validation set, then refit best model on train only ---
    print("Training Bayesian (Gaussian NB)...")
    estimator, scaler, best_params = train_bayesian(X_train, y_train, X_val, y_val, config=model_cfg)

    # --- Evaluate on the test set (never used during training or model selection) ---
    # Apply the same scaling that was fit on the training set, then predict
    # (each test sample is assigned the class with highest posterior probability)
    X_test_s = scaler.transform(X_test)
    y_pred = estimator.predict(X_test_s)

    # --- Compute metrics and save outputs ---
    metrics = classification_metrics(y_test, y_pred, labels=ACTION_NAMES)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    plot_confusion_matrix(
        y_test, y_pred, ACTION_NAMES,
        save_path=results_dir / "confusion_matrix.png",
        title="Bayesian (Gaussian NB) - Test Set",
    )
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Results saved to", results_dir)


if __name__ == "__main__":
    main()
