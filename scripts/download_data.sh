#!/bin/bash
# Download KTH dataset from Kaggle into data/raw.
# Requires: pip install kaggle and ~/.kaggle/kaggle.json
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle datasets download vafaeii/kth-action-recognition-dataset -p data/raw --unzip
echo "Dataset ready in data/raw/"
