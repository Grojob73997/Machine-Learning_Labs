# TASK 1 - DATASET
from pathlib import Path
from decimal import Decimal, localcontext
from importlib import metadata
import platform
import csv
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

# Reproducibility. PyTorch is used only for the course-required seed calls.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Use one BLAS thread for consistent small-matrix timing.
blas_limit = threadpool_limits(limits=1, user_api="blas")

# This file belongs in lab02/src.
LAB = Path(__file__).resolve().parent.parent
RESULTS = LAB / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Record Python and installed package versions.
versions = [f"Python: {platform.python_version()}"]
versions += sorted(
    f"{d.metadata['Name']}=={d.version}"
    for d in metadata.distributions()
)
version_text = "\n".join(versions)
print(version_text)
(RESULTS / "today_versions.txt").write_text(
    version_text, encoding="utf-8"
)


def save_csv(filename, rows):
    with (RESULTS / filename).open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# Load images and labels.
X, y = load_digits(return_X_y=True)

# Pixel values originally range from 0 to 16.
X = X.astype(np.float64) / 16.0

# Stratified 70/30 split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Convert labels to one-hot vectors.
Y_train = np.eye(10)[y_train]
Y_test = np.eye(10)[y_test]

print("\nTASK 1")
print("Training:", X_train.shape, Y_train.shape)
print("Testing: ", X_test.shape, Y_test.shape)