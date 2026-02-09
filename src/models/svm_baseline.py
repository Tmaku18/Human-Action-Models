"""
SVM baseline: standardize features, grid search on val, evaluate on test.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from sklearn.model_selection import GridSearchCV
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
    Grid search SVM on (X_train, y_train) with validation scoring, or use val set explicitly.
    Returns (best_estimator, scaler, best_params).
    """
    if config is None:
        config = load_svm_config(config_path) if config_path else {}
    random_state = config.get("random_state", 42)
    param_grid = config.get("param_grid", {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]})

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    base = SVC(
        kernel=config.get("kernel", "rbf"),
        decision_function_shape=config.get("decision_function_shape", "ovr"),
        random_state=random_state,
    )
    gs = GridSearchCV(
        base,
        param_grid,
        cv=3,
        scoring="balanced_accuracy",
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train_s, y_train)
    best_estimator = gs.best_estimator_
    return best_estimator, scaler, gs.best_params_
