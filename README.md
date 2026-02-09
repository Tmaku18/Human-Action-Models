# Human Action Classification (KTH Dataset)

CSC 8850 Advanced Machine Learning project: classify human actions from video using a classical ML pipeline (SVM, Bayesian, k-NN) on the KTH Action Recognition dataset.

## Overview

- **Goal:** Accurately classify six human actions from video using spatiotemporal features and classical classifiers.
- **Dataset:** [KTH Action Recognition](https://www.csc.kth.se/cvap/actions/) — 2,391 sequences, 25 subjects, 4 scenarios (s1–s4), 6 actions (walking, jogging, running, boxing, hand waving, hand clapping), 25 fps, 160×120 px. Official mirror: [Kaggle](https://www.kaggle.com/datasets/vafaeii/kth-action-recognition-dataset/data).
- **Evaluation:** Subject-based split (70% train / 15% val / 15% test), balanced accuracy, macro F1, confusion matrices.

## Setup

```bash
cd Human_Movement_SVM
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

## How to Run

1. **Download KTH** (once):
   - **Kaggle auth:** Use either (a) `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows), or (b) set `KAGGLE_API_TOKEN` to your token (from Kaggle → Settings → API → Create New Token). Do not commit tokens or kaggle.json; if a token was exposed, revoke it in Kaggle and create a new one.
   - **Python (any OS):** `python scripts/download_data.py` (uses kagglehub).
   - **Bash (Linux/Mac):** `export KAGGLE_API_TOKEN=your_token` then `bash scripts/download_data.sh`, or use kaggle.json so `kaggle competitions list` / `kaggle datasets download ...` work.
   - Or download from [Kaggle](https://www.kaggle.com/datasets/vafaeii/kth-action-recognition-dataset/data) and unzip into `data/raw/`.

2. **Preprocess and extract features** (builds metadata, subject-based splits, caches HOF features):
   ```bash
   python scripts/run_preprocess.py --config configs/data.yaml
   ```

3. **Train models** (each reads cached features and writes to `results/<model>/`):
   ```bash
   python scripts/train_svm.py
   python scripts/train_bayesian.py
   python scripts/train_knn.py
   ```

4. **Compare all models**:
   ```bash
   python scripts/evaluate_all.py
   ```

Configs: `configs/data.yaml` (paths, n_frames, feature params), `configs/model_svm.yaml`, `configs/model_bayesian.yaml`, `configs/model_knn.yaml`.

## Implementation Plan

Detailed tasks and phases are in [`.cursor/plans/human-action-classification-implementation.md`](.cursor/plans/human-action-classification-implementation.md). Summary:

1. **Environment & data** — Repo structure, dependencies, KTH download, subject-based splits.
2. **Preprocessing & features** — Decode AVI → grayscale, uniform frame sampling, spatiotemporal feature extraction, caching.
3. **Baseline (SVM)** — Train/val/test pipeline, hyperparameter tuning, metrics (target: 02/08).
4. **Other models** — Bayesian classifier, k-NN (target: 02/28).
5. **Analysis & improvement** — Error analysis, improvements, final evaluation (03/09–03/12).
6. **Report & deliverables** — Clean code, README, report, presentation (03/18).

## Project Layout

```
├── configs/          # data.yaml, model_*.yaml
├── data/
│   ├── raw/          # KTH from Kaggle
│   ├── processed/    # Cached features (*.npz), action_names.json
│   └── splits/       # train.txt, val.txt, test.txt
├── src/
│   ├── data/         # dataset, split, preprocess
│   ├── features/     # HOF spatiotemporal extraction
│   ├── models/       # SVM, Bayesian, k-NN
│   └── evaluation/   # balanced accuracy, F1, confusion matrix
├── scripts/          # download_data, run_preprocess, train_*, evaluate_all
├── results/          # baseline_svm/, bayesian/, knn/ (metrics, plots)
└── requirements.txt
```

## References

- **Official dataset:** [Recognition of human actions](https://www.csc.kth.se/cvap/actions/) (KTH/CVAP). File naming: `personXX_action_dY_uncomp.avi` (e.g. `person15_walking_d1_uncomp.avi`). Official split: 8 train / 8 val / 9 persons (see [00sequences.txt](https://www.csc.kth.se/cvap/actions/) for sequence boundaries).
- Schuldt, C., Laptev, I., & Caputo, B. (2004). Recognizing human actions: A local SVM approach. *ICPR 2004*. https://doi.org/10.1109/icpr.2004.1334462
