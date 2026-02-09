"""
k-NN classifier with optional grid search (n_neighbors, metric).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def load_knn_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str | Path] = None,
) -> tuple:
    """
    StandardScaler + k-NN. Grid search on n_neighbors and metric.
    Returns (best_estimator, scaler, best_params).
    """
    if config is None:
        config = load_knn_config(config_path) if config_path else {}
    random_state = config.get("random_state", 42)
    param_grid = config.get("param_grid", {"n_neighbors": [3, 5, 7, 11], "metric": ["euclidean", "manhattan"]})

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    base = KNeighborsClassifier(weights=config.get("weights", "uniform"))
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
