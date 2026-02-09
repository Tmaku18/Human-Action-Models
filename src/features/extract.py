"""
Spatiotemporal features: dense optical flow + block-based HOF histograms,
aggregated over time to one vector per video.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _optical_flow_magnitude_angle(prev: np.ndarray, curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dense optical flow (Farneback); return magnitude and angle (radians)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    return mag.astype(np.float32), ang.astype(np.float32)


def _hof_block(mag: np.ndarray, ang: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """Histogram of optical flow for one block (single frame pair)."""
    h, w = mag.shape
    hist, _ = np.histogram(ang.ravel(), bins=n_bins, range=(-np.pi, np.pi), weights=mag.ravel())
    return hist.astype(np.float32)


def _frame_pair_hof_grid(
    prev: np.ndarray,
    curr: np.ndarray,
    n_blocks_h: int,
    n_blocks_w: int,
    n_bins: int = 9,
) -> np.ndarray:
    """Divide frame into grid, compute HOF per block; return flat vector."""
    mag, ang = _optical_flow_magnitude_angle(prev, curr)
    h, w = mag.shape
    bh, bw = h // n_blocks_h, w // n_blocks_w
    feats = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            y1, y2 = i * bh, (i + 1) * bh if i < n_blocks_h - 1 else h
            x1, x2 = j * bw, (j + 1) * bw if j < n_blocks_w - 1 else w
            m = mag[y1:y2, x1:x2]
            a = ang[y1:y2, x1:x2]
            feats.append(_hof_block(m, a, n_bins))
    return np.concatenate(feats)


def extract_features(
    frames: np.ndarray,
    n_blocks_h: int = 5,
    n_blocks_w: int = 5,
    n_bins: int = 9,
) -> np.ndarray:
    """
    frames: (T, H, W) grayscale. Compute HOF grid for each consecutive pair, then aggregate.
    Returns one vector: mean and max over time of per-pair HOF vectors (so 2 * n_blocks_h * n_blocks_w * n_bins).
    """
    if frames is None or len(frames) < 2:
        return np.array([])
    block_dim = n_blocks_h * n_blocks_w * n_bins
    per_pair = []
    for t in range(len(frames) - 1):
        v = _frame_pair_hof_grid(
            frames[t], frames[t + 1],
            n_blocks_h=n_blocks_h, n_blocks_w=n_blocks_w, n_bins=n_bins,
        )
        per_pair.append(v)
    stack = np.stack(per_pair, axis=0)
    mean_vec = np.mean(stack, axis=0)
    max_vec = np.max(stack, axis=0)
    return np.concatenate([mean_vec, max_vec]).astype(np.float32)


def extract_features_batch(
    frame_list: List[np.ndarray],
    n_blocks_h: int = 5,
    n_blocks_w: int = 5,
    n_bins: int = 9,
    verbose: bool = False,
) -> np.ndarray:
    """Extract feature vector for each video (list of frame arrays). Returns (N, D)."""
    feats = []
    it = frame_list
    if verbose:
        try:
            from tqdm import tqdm
            it = tqdm(frame_list, desc="Extracting features")
        except ImportError:
            pass
    for frames in it:
        v = extract_features(frames, n_blocks_h=n_blocks_h, n_blocks_w=n_blocks_w, n_bins=n_bins)
        if len(v) > 0:
            feats.append(v)
    return np.stack(feats, axis=0) if feats else np.empty((0, 0))
