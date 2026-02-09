"""
Bayesian classifier: Gaussian Naive Bayes with optional grid search (var_smoothing).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def load_bayesian_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_bayesian(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str | Path] = None,
) -> tuple:
    """
    StandardScaler + GaussianNB. Optional grid search on var_smoothing.
    Returns (best_estimator, scaler, best_params).
    """
    if config is None:
        config = load_bayesian_config(config_path) if config_path else {}
    random_state = config.get("random_state", 42)
    param_grid = config.get("param_grid", {"var_smoothing": [1e-9, 1e-8, 1e-7]})
    # Ensure var_smoothing are floats (YAML may load as str)
    if "var_smoothing" in param_grid:
        param_grid = {**param_grid, "var_smoothing": [float(x) for x in param_grid["var_smoothing"]]}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    base = GaussianNB()
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
    return gs.best_estimator_, scaler, gs.best_params_
