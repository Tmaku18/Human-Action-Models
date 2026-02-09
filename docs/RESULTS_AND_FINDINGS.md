# Results and Findings

Test-set results from the KTH human-action classification pipeline (subject-based 70/15/15 split, dense HOF features, validation-set tuning). For pipeline and model details, see [PIPELINE_AND_MODEL.md](PIPELINE_AND_MODEL.md).

---

## 1. Our results (test set)

| Model | Balanced accuracy | Macro F1 | Accuracy |
|-------|-------------------|----------|----------|
| **SVM (RBF)** | 0.600 | 0.602 | 0.60 |
| **Bayesian (Gaussian NB)** | 0.45 | 0.45 | 0.45 |
| **k-NN** | 0.39 | 0.39 | 0.39 |

- **Classes**: walking, jogging, running, boxing, handwaving, handclapping (6 actions).
- **Best hyperparameters** (e.g. SVM): saved in `results/baseline_svm/best_params.json`, `results/bayesian/best_params.json`, `results/knn/best_params.json`.
- **Confusion matrices**: `results/*/confusion_matrix.png`.

---

## 2. What the metrics mean (plain English)

- **Accuracy**: Fraction of test videos where the predicted action matches the true action. Can be misleading when some actions are much more frequent than others.

- **Balanced accuracy**: For each of the 6 actions we take “what fraction of true X did we correctly label as X?” (recall per class), then average those six numbers. Each class is weighted equally, so rare or hard actions count as much as common ones. **0.60** means we’re right about 60% of the time when each class is treated equally.

- **Macro F1**: For each action we compute F1 (harmonic mean of precision and recall), then average over the 6 actions. Again every class counts equally. Good when you care about every action, not just the most frequent.

We report balanced accuracy and macro F1 so that small or difficult classes (e.g. handclapping) are not drowned out by easier or more frequent ones (e.g. walking).

---

## 3. Comparison to published work (KTH)

Reported results on the same KTH dataset (6 actions, subject-based or official split):

| Setting | Reported performance | Notes |
|--------|----------------------|--------|
| **Our SVM (dense HOF, RBF)** | **~60%** | Simplified features (no interest points), 70/15/15 split, validation-based tuning. |
| **Our Bayesian** | **~45%** | Same features; Gaussian NB. |
| **Our k-NN** | **~39%** | Same features; distance-based. |
| **Schuldt et al. (2004)** | Baseline | Local SVM with space-time interest points and local descriptors (HOG/HOF). |
| **Laptev et al. (space-time interest points, improved)** | **~91.8%** | Strong hand-crafted features, multi-channel SVM. |
| **Later work (e.g. contextual space-time features)** | **~96%** | Refined hand-crafted or hybrid pipelines. |
| **Deep learning (e.g. CNN-GRU)** | Often **90%+** | Learned representations. |

**Takeaway**: Our **~60%** SVM result is reasonable for a **simplified classical pipeline** (dense HOF + mean/max, no interest-point detection, no codebook). It is below **~92–96%** from stronger hand-crafted (interest-point + descriptor) or deep-learning methods. The gap is expected because we use a cheaper, easier-to-implement feature design (Option A) rather than the full paper-style pipeline (Option B) or learned features. Reaching 90%+ would require adopting space-time interest points and better descriptors or moving to a deep-learning model.

---

## 4. References

- Pipeline and training: [PIPELINE_AND_MODEL.md](PIPELINE_AND_MODEL.md)
- Schuldt et al. (2004): [IEEE 1334462](https://ieeexplore.ieee.org/document/1334462)
- Raw metrics: `results/baseline_svm/metrics.json`, `results/bayesian/metrics.json`, `results/knn/metrics.json`
