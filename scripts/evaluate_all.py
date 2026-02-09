"""
Print comparison table of all models (SVM, Bayesian, k-NN) from results/*/metrics.json.
Usage: python scripts/evaluate_all.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ["baseline_svm", "bayesian", "knn"]


def main():
    rows = []
    for name in MODELS:
        p = RESULTS / name / "metrics.json"
        if not p.exists():
            rows.append((name, None, None, "(run train script first)"))
            continue
        with open(p) as f:
            m = json.load(f)
        rows.append((
            name,
            m.get("balanced_accuracy"),
            m.get("macro_f1"),
            "",
        ))
    print("Model            | Balanced Acc | Macro F1")
    print("-" * 50)
    for name, ba, f1, note in rows:
        if ba is not None:
            print(f"{name:16} | {ba:.4f}        | {f1:.4f}")
        else:
            print(f"{name:16} | {note}")


if __name__ == "__main__":
    main()
