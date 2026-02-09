"""
Video decoding: load AVI, grayscale, uniform sampling to fixed number of frames.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_video_frames(
    path: str | Path,
    n_frames: int = 16,
    grayscale: bool = True,
    target_size: Optional[tuple] = None,
) -> np.ndarray:
    """
    Load video and return array of shape (n_frames, H, W) or (n_frames, H, W, 1).
    Uniformly sample n_frames over time. Missing/corrupt files return empty array.
    """
    path = Path(path)
    if not path.exists():
        return np.array([])

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.array([])

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.array([])

    # Uniform sampling indices
    indices = np.linspace(0, total - 1, num=n_frames, dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if target_size:
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frames.append(frame)

    cap.release()
    if len(frames) != n_frames:
        return np.array([])
    return np.stack(frames, axis=0)


def load_video_frames_batch(
    paths: list,
    n_frames: int = 16,
    grayscale: bool = True,
    target_size: Optional[tuple] = None,
    verbose: bool = False,
):
    """Load multiple videos; yield (path, frames) for each. Skip failed loads."""
    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="Loading videos") if verbose else paths
    except ImportError:
        iterator = paths

    for path in iterator:
        frames = load_video_frames(path, n_frames=n_frames, grayscale=grayscale, target_size=target_size)
        if len(frames) > 0:
            yield path, frames
