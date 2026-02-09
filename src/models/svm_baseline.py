"""
SVM baseline: standardize features, grid search on validation set, evaluate on test.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_svm_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str | Path] = None,
) -> tuple:
    """
    Fit StandardScaler on train; scale train/val.
    Grid search SVM using the validation set for model selection (PredefinedSplit).
    The best estimator is refit on train only before return (val used only for tuning).
    Returns (best_estimator, scaler, best_params).
    """
    if config is None:
        config = load_svm_config(config_path) if config_path else {}
    random_state = config.get("random_state", 42)
    param_grid = config.get("param_grid", {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]})

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Merge train and val for GridSearchCV; use PredefinedSplit so only val is the "test" fold
    n_train, n_val = len(X_train_s), len(X_val_s)
    X_dev = np.vstack([X_train_s, X_val_s])
    y_dev = np.concatenate([y_train, y_val])
    test_fold = np.array([-1] * n_train + [0] * n_val)
    cv = PredefinedSplit(test_fold)

    base = SVC(
        kernel=config.get("kernel", "rbf"),
        decision_function_shape=config.get("decision_function_shape", "ovr"),
        random_state=random_state,
    )
    gs = GridSearchCV(
        base,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_dev, y_dev)
    # Refit best estimator on train only (val was only for selection; keeps test evaluation unbiased)
    best_estimator = clone(gs.best_estimator_)
    best_estimator.fit(X_train_s, y_train)
    return best_estimator, scaler, gs.best_params_
