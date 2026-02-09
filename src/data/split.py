"""
Subject-based train/val/test split for KTH (no identity leakage).
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def make_subject_splits(
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split subject IDs into train/val/test; return row indices for each split.
    All clips of a subject go to the same split.
    """
    subject_ids = metadata["subject_id"].unique()
    n = len(subject_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subject_ids)

    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0

    train_subjects = set(perm[:n_train])
    val_subjects = set(perm[n_train : n_train + n_val])
    test_subjects = set(perm[n_train + n_val :])

    train_idx = metadata.index[metadata["subject_id"].isin(train_subjects)].to_numpy()
    val_idx = metadata.index[metadata["subject_id"].isin(val_subjects)].to_numpy()
    test_idx = metadata.index[metadata["subject_id"].isin(test_subjects)].to_numpy()

    return train_idx, val_idx, test_idx


def save_splits(
    splits_dir: str | Path,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Save split file lists (paths) and subject lists for reproducibility."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        paths = metadata.loc[idx, "video_path"].tolist()
        (splits_dir / f"{name}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    # Save subject assignment
    subj = []
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        subj.extend(metadata.loc[idx, "subject_id"].unique().tolist())
    # one line per subject: subject_id,split
    lines = []
    for _, row in metadata.drop_duplicates("subject_id").iterrows():
        sid = row["subject_id"]
        if sid in metadata.loc[train_idx, "subject_id"].values:
            lines.append(f"{sid},train")
        elif sid in metadata.loc[val_idx, "subject_id"].values:
            lines.append(f"{sid},val")
        else:
            lines.append(f"{sid},test")
    (splits_dir / "subjects_splits.txt").write_text("\n".join(sorted(set(lines))) + "\n", encoding="utf-8")


def load_splits(splits_dir: str | Path) -> Tuple[List[str], List[str], List[str]]:
    """Load train/val/test path lists from data/splits/*.txt."""
    splits_dir = Path(splits_dir)
    def read_paths(name: str) -> List[str]:
        p = splits_dir / f"{name}.txt"
        if not p.exists():
            return []
        return [s.strip() for s in p.read_text(encoding="utf-8").strip().splitlines() if s.strip()]

    return read_paths("train"), read_paths("val"), read_paths("test")
