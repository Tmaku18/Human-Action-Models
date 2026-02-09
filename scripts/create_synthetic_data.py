"""
Create minimal synthetic KTH-like AVI files for pipeline testing when Kaggle is not configured.
Writes into data/raw/ with naming personXX_action_dY_synth.avi (parsed by dataset.py).
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
ACTION_NAMES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
N_SUBJECTS = 10   # enough for 70/15/15 to have train/val/test all non-empty
N_SCENARIOS = 2
FPS = 25
SECONDS = 2
W, H = 160, 120


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    count = 0
    for person in range(1, N_SUBJECTS + 1):
        for action in ACTION_NAMES:
            for scenario in range(1, N_SCENARIOS + 1):
                path = DATA_RAW / f"person{person:02d}_{action}_d{scenario}_synth.avi"
                out = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
                n_frames = FPS * SECONDS
                for i in range(n_frames):
                    # Slight variation per frame so optical flow is non-zero
                    frame = np.random.randint(0, 256, (H, W), dtype=np.uint8)
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)
                    out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                out.release()
                count += 1
    print(f"Created {count} synthetic AVI files in {DATA_RAW}")


if __name__ == "__main__":
    main()
