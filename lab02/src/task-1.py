# G M Robi Farhan Rojob | 73997 | Lab 2
# TASK 1 - DATASET
from pathlib import Path
from decimal import Decimal, localcontext
import csv
import random
import time
import platform
from importlib import metadata

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save plots to PNG without requiring a separate plot window.
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
blas_limit = threadpool_limits(limits=1, user_api="blas")

LAB = Path(__file__).resolve().parent.parent
RESULTS = LAB / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
versions = [f"Python: {platform.python_version()}"]
versions += sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
version_text = "\n".join(versions)
print(version_text)
(RESULTS / "today_versions.txt").write_text(version_text, encoding="utf-8")

def save_csv(filename, rows):
    with (RESULTS / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

X, y = load_digits(return_X_y=True)
X = X.astype(np.float64) / 16.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
Y_train = np.eye(10)[y_train]
Y_test = np.eye(10)[y_test]
print("TASK 1")
print("Training:", X_train.shape, Y_train.shape)
print("Testing: ", X_test.shape, Y_test.shape)

