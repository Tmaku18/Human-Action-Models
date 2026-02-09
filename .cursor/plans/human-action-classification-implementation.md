# Human Action Classification – Implementation Plan

Based on: **CSC 8850 - Donggeun Yoo & Quan Do - Project Proposal** (KTH dataset, SVM/Bayesian/k-NN pipeline).

---

## 1. Project Overview

| Item | Description |
|------|-------------|
| **Goal** | Classify human actions from video using a classical ML pipeline (no deep learning). |
| **Dataset** | KTH Action Recognition ([Kaggle](https://www.kaggle.com/datasets/vafaeii/kth-action-recognition-dataset/data)) |
| **Actions** | 6 classes: walking, jogging, running, boxing, hand waving, hand clapping |
| **Scale** | 2,391 videos, 25 subjects, 4 scenarios (s1–s4), 25 fps, 160×120 px |
| **Evaluation** | Subject-based split (70% train / 15% val / 15% test), balanced accuracy, macro F1, confusion matrices |

---

## 2. Implementation Phases

### Phase 1: Environment & Data (Priority: High)

- [x] **1.1 Repo structure**
  - `data/` – raw and processed data (gitignored or DVC)
  - `src/` – preprocessing, features, models, evaluation
  - `notebooks/` or `scripts/` – experiments and runs
  - `configs/` – YAML/JSON for paths, splits, hyperparameters
  - `requirements.txt` or `environment.yml` (Python 3.8+)

- [x] **1.2 Dependencies**
  - Video I/O: `opencv-python` (or `av`)
  - Numerics/ML: `numpy`, `scipy`, `scikit-learn`
  - SVM: `scikit-learn` (or optional `libsvm` for large scale)
  - Optional: `pandas` for metadata, `tqdm` for progress, `pyyaml` for configs

- [x] **1.3 Dataset**
  - Download KTH from Kaggle (script or manual) into `data/raw/` or equivalent
  - Parse directory structure to get: subject id, scenario (s1–s4), action label, video path
  - Build a **metadata table**: `[video_path, subject_id, scenario, action]` for reproducible splits

- [x] **1.4 Subject-based splits**
  - Implement split as in Schuldt et al.: 70% / 15% / 15% of **subjects** (not clips)
  - Assign subject IDs to `train / val / test`; all clips of a subject go to one split
  - Save split indices or file lists (e.g. `data/splits/train.txt`, `val.txt`, `test.txt`) for reproducibility

---

### Phase 2: Preprocessing & Feature Extraction (Priority: High)

- [x] **2.1 Video decoding**
  - Load AVI with OpenCV; convert frames to **grayscale** to reduce dimensionality and cost
  - Handle missing/corrupt files gracefully (log and skip or flag)

- [x] **2.2 Temporal normalization**
  - **Uniform sampling**: for each video, sample a **fixed number of frames** (e.g. 16 or 24) uniformly over time
  - Same length per sequence so downstream features have consistent dimensions

- [x] **2.3 Spatiotemporal features**
  - Choose a concrete scheme (proposal mentions “spatiotemporal features”):
    - **Option A (simpler):** Histogram of Optical Flow (HOF) + block-based histograms per frame, then aggregate (e.g. mean/max over time)
    - **Option B (Schuldt-style):** Space-time interest points + local descriptors (e.g. HOG/HOF around interest points), then bag-of-words
  - Output: one **feature vector per video** (fixed dimension)
  - Implement in a modular way so the classifier stage only sees numpy arrays `(n_samples, n_features)` and labels

- [x] **2.4 Caching**
  - Save extracted features and labels (e.g. `data/processed/features_train.npz`) so training does not recompute from video every time
  - Include in cache key: split name, number of frames, feature config version

---

### Phase 3: Baseline Model – SVM (Priority: High)

- [x] **3.1 Training**
  - Use **subject-based train** set only; **validation** set for model selection
  - Standardize features (e.g. `StandardScaler` fit on train, apply to val/test)
  - Train multi-class SVM (e.g. `SVC(kernel='rbf', decision_function_shape='ovr')` or `'ovo'`)
  - Reference: Schuldt et al. – “local SVM approach” on similar features

- [x] **3.2 Hyperparameter tuning**
  - Use **validation set** to tune: `C`, `gamma` (for RBF), and optionally kernel type
  - Method: grid search or random search (e.g. `GridSearchCV` with fixed val split to avoid overfitting to val)

- [x] **3.3 Evaluation**
  - Predict on **test set** (subject-disjoint)
  - Compute: **balanced accuracy**, **macro F1**, **confusion matrix**
  - Save metrics and plots (e.g. `results/baseline_svm/metrics.json`, `confusion_matrix.png`)

- [x] **3.4 Reproducibility**
  - Fix random seeds; document preprocessing + feature version and split IDs so baseline is reproducible

**Target date (from proposal):** 02/08 – baseline model done.

---

### Phase 4: Alternative Models (Priority: Medium)

- [x] **4.1 Bayesian classifier**
  - Implement a probabilistic classifier (e.g. Gaussian Naive Bayes, or LDA/QDA for multi-class)
  - Same train/val/test and feature pipeline as SVM
  - Tune any hyperparameters (e.g. smoothing) on validation set
  - Report balanced accuracy, macro F1, confusion matrix

- [x] **4.2 Distance-based classifier**
  - Implement either:
    - **k-NN**, or
    - **Minimum Distance Classifier** (e.g. class centroids, assign to nearest)
  - For k-NN: tune `k` and distance metric on validation set
  - Same evaluation metrics and test set

- [x] **4.3 Comparison**
  - Summary table: SVM vs Bayesian vs k-NN (or MDC) on test set (balanced accuracy, macro F1)
  - Short analysis: which classes are confused, effect of scenarios (s1–s4) if you log scenario in metadata

**Target date (from proposal):** 02/28 – other models done.

---

### Phase 5: Analysis & Improvement (Priority: Medium)

- [ ] **5.1 Error analysis**
  - Use confusion matrices and (if available) per-scenario or per-subject breakdowns
  - Identify: confused action pairs, hard scenarios, possible data issues

- [ ] **5.2 Improvement ideas (by 03/09)**
  - Feature: different temporal sampling, more robust optical flow, or different descriptors
  - Model: kernel choice, calibration for SVM; prior for Bayesian; weighted k-NN
  - Data: exclude ambiguous clips or balance classes if needed
  - Pick one or two improvements and implement

- [ ] **5.3 Final evaluation**
  - Re-run evaluation on test set for improved pipeline; report same metrics
  - Keep baseline numbers for comparison in report

**Target dates:** 03/09 analysis; 03/12 improvement done.

---

### Phase 6: Report & Deliverables (Priority: Before deadline)

- [x] **6.1 Code**
  - Clean, documented code; README with setup, data download, and how to run preprocessing → train → evaluate
  - Config or CLI to reproduce results (e.g. `python scripts/train_svm.py --model-config configs/model_svm.yaml`)

- [ ] **6.2 Report & presentation**
  - Finalize report and slides (proposal: 03/18)
  - Include: dataset summary, methodology (preprocessing, features, splits, models), results (tables + figures), discussion, references (Schuldt et al.)

---

## 3. Suggested Project Layout

```text
Human_Movement_SVM/
├── .cursor/
│   └── plans/
│       └── human-action-classification-implementation.md   # this file
├── configs/
│   ├── data.yaml           # paths, split ratios, frame count
│   └── model_*.yaml        # per-model hyperparameters
├── data/
│   ├── raw/                # KTH from Kaggle
│   ├── processed/          # features, labels, cache
│   └── splits/             # train/val/test subject or file lists
├── src/
│   ├── data/
│   │   ├── dataset.py      # load metadata, list clips
│   │   ├── split.py        # subject-based split
│   │   └── preprocess.py   # decode video, grayscale, sample frames
│   ├── features/
│   │   └── extract.py      # spatiotemporal feature extraction
│   ├── models/
│   │   ├── svm_baseline.py
│   │   ├── bayesian.py
│   │   └── knn_or_mdc.py
│   └── evaluation/
│       └── metrics.py      # balanced accuracy, F1, confusion matrix
├── scripts/ or notebooks/
│   ├── download_data.py
│   ├── run_preprocess.py
│   ├── train_svm.py
│   ├── train_bayesian.py
│   ├── train_knn.py
│   └── evaluate_all.py
├── results/                # metrics, plots, checkpoints
├── requirements.txt
└── README.md
```

---

## 4. Success Criteria (from proposal)

- Subject-based splits used consistently; no identity leakage.
- Baseline SVM implemented and comparable in setup to prior work (Schuldt et al.).
- At least two other classifiers (Bayesian + distance-based) implemented and compared.
- Metrics: balanced accuracy, macro F1, confusion matrices reported on the **test set**.
- Reproducible runs via config/scripts and documented environment.

---

## 5. References

- **Official dataset:** [Recognition of human actions](https://www.csc.kth.se/cvap/actions/) (KTH/CVAP). 2391 sequences, 25×6×4=600 AVI files, `personXX_action_dY_uncomp.avi`; subdivision in 00sequences.txt; official split 8/8/9 persons (train/val/test).
- Schuldt, C., Laptev, I., & Caputo, B. (2004). Recognizing human actions: A local SVM approach. *Proceedings of the 17th International Conference on Pattern Recognition (ICPR)*. https://doi.org/10.1109/icpr.2004.1334462
- Kaggle mirror: https://www.kaggle.com/datasets/vafaeii/kth-action-recognition-dataset/data
