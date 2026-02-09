"""
Build metadata, subject-based splits, extract features, and cache to data/processed.
Usage: python scripts/run_preprocess.py [--config configs/data.yaml] [--workers N]

Use --workers to parallelize video loading and feature extraction (e.g. --workers 8 or -1 for all cores).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from src.data.dataset import load_metadata, ACTION_NAMES
from src.data.split import make_subject_splits, save_splits
from src.data.preprocess import load_video_frames_batch, load_video_frames
from src.features.extract import extract_features_batch, extract_features


def _load_and_extract_one(
    path,
    n_frames: int,
    grayscale: bool,
    n_blocks_h: int,
    n_blocks_w: int,
    n_bins: int,
):
    """Load one video and extract HOF features. Returns (path, feature_vector or None)."""
    frames = load_video_frames(path, n_frames=n_frames, grayscale=grayscale)
    if frames is None or len(frames) < 2:
        return (str(path), None)
    feats = extract_features(
        frames, n_blocks_h=n_blocks_h, n_blocks_w=n_blocks_w, n_bins=n_bins
    )
    return (str(path), feats)


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "data.yaml")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for feature extraction (1=sequential, -1=all cores). Default: 1.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = ROOT / cfg["data_dir"]
    processed_dir = ROOT / cfg["processed_dir"]
    splits_dir = ROOT / cfg["splits_dir"]
    n_frames = cfg["n_frames"]
    grayscale = cfg.get("grayscale", True)
    n_blocks_h = cfg.get("n_blocks_h", 5)
    n_blocks_w = cfg.get("n_blocks_w", 5)
    hof_bins = cfg.get("hof_bins", 9)
    cache_features = cfg.get("cache_features", True)
    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    test_ratio = cfg["test_ratio"]
    split_seed = cfg["split_seed"]
    workers = args.workers

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1) Metadata
    print("Loading metadata...")
    metadata = load_metadata(data_dir)
    if metadata.empty:
        print("No AVI files found under", data_dir)
        print("Download KTH first: python scripts/download_data.py")
        raise SystemExit(1)
    print("Videos:", len(metadata), "| Subjects:", metadata["subject_id"].nunique())

    # 2) Splits
    train_idx, val_idx, test_idx = make_subject_splits(
        metadata, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=split_seed,
    )
    save_splits(splits_dir, metadata, train_idx, val_idx, test_idx)
    metadata["split"] = "train"
    metadata.loc[val_idx, "split"] = "val"
    metadata.loc[test_idx, "split"] = "test"

    # 3) Extract features per split and cache
    feature_version = cfg.get("feature_version", "v1")
    cache_key = f"n{n_frames}_b{n_blocks_h}x{n_blocks_w}_hof{hof_bins}_{feature_version}"

    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        paths = metadata.loc[idx, "video_path"].tolist()
        labels = metadata.loc[idx, "action_id"].values
        cache_file = processed_dir / f"features_{split_name}_{cache_key}.npz"
        if cache_features and cache_file.exists():
            print(f"Loading cached {split_name}...")
            data = np.load(cache_file)
            X = data["X"]
            y = data["y"]
        else:
            print(f"Extracting features for {split_name} ({len(paths)} videos, workers={workers})...")
            path_to_label = dict(zip(metadata.loc[idx, "video_path"].tolist(), labels))
            if workers is None or workers == 1:
                frames_list = []
                labels_ok = []
                for path, frames in load_video_frames_batch(
                    paths, n_frames=n_frames, grayscale=grayscale, verbose=True
                ):
                    frames_list.append(frames)
                    labels_ok.append(path_to_label[str(path)])
                X = extract_features_batch(
                    frames_list,
                    n_blocks_h=n_blocks_h,
                    n_blocks_w=n_blocks_w,
                    n_bins=hof_bins,
                    verbose=True,
                )
                y = np.array(labels_ok, dtype=np.int64)
            else:
                n_jobs = workers if workers > 0 else -1
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_load_and_extract_one)(
                        p, n_frames, grayscale, n_blocks_h, n_blocks_w, hof_bins
                    )
                    for p in paths
                )
                result_by_path = {r[0]: r[1] for r in results}
                features_list = []
                labels_ok = []
                for p in paths:
                    feats = result_by_path.get(str(p))
                    if feats is not None:
                        features_list.append(feats)
                        labels_ok.append(path_to_label[str(p)])
                X = np.stack(features_list, axis=0) if features_list else np.empty((0, 0))
                y = np.array(labels_ok, dtype=np.int64)
            if cache_features:
                np.savez_compressed(cache_file, X=X, y=y)
        print(f"  {split_name}: X {X.shape}, y {y.shape}")

    # Save label names for evaluation
    (processed_dir / "action_names.json").write_text(json.dumps(ACTION_NAMES), encoding="utf-8")
    print("Preprocess done. Cached under", processed_dir)


if __name__ == "__main__":
    main()
