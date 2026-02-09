"""
Download KTH dataset from Kaggle into data/raw using kagglehub.
Requires: pip install kagglehub and Kaggle API credentials (same as kaggle CLI).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


def main():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("vafaeii/kth-action-recognition-dataset")
    print("Path to dataset files:", path)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    src = Path(path)
    # Copy contents into data/raw so the rest of the pipeline finds it
    for item in src.iterdir():
        dest = DATA_RAW / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print("Dataset copied to", DATA_RAW)


if __name__ == "__main__":
    main()
