"""
Train k-NN classifier. Same pipeline as SVM.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.data.dataset import ACTION_NAMES
from src.models.knn_or_mdc import train_knn, load_knn_config
from src.evaluation.metrics import classification_metrics, plot_confusion_matrix


def load_data_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_cached_features(processed_dir: Path, data_cfg: dict):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=Path, default=ROOT / "configs" / "data.yaml")
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model_knn.yaml")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "knn")
    args = ap.parse_args()

    data_cfg = load_data_config(args.data_config)
    model_cfg = load_knn_config(args.model_config)
    processed_dir = ROOT / data_cfg["processed_dir"]
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    data = get_cached_features(processed_dir, data_cfg)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    print("Training k-NN...")
    estimator, scaler, best_params = train_knn(X_train, y_train, X_val, y_val, config=model_cfg)
    X_test_s = scaler.transform(X_test)
    y_pred = estimator.predict(X_test_s)

    metrics = classification_metrics(y_test, y_pred, labels=ACTION_NAMES)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    plot_confusion_matrix(
        y_test, y_pred, ACTION_NAMES,
        save_path=results_dir / "confusion_matrix.png",
        title="k-NN - Test Set",
    )
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Results saved to", results_dir)


if __name__ == "__main__":
    main()
