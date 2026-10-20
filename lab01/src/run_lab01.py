"""Run Lab 1 measurements and write reproducible evidence files."""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab01" / "results"


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def timed_call(function: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    function()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def median_training_time(factory: Callable[[], Any], x_train: np.ndarray, y_train: np.ndarray) -> tuple[float, list[float]]:
    # One warm-up followed by the required five measured runs.
    factory().fit(x_train, y_train)
    runs = [timed_call(lambda: factory().fit(x_train, y_train)) for _ in range(5)]
    return statistics.median(runs), runs


def median_inference_time(model: Any, sample: np.ndarray) -> tuple[float, list[float]]:
    model.predict(sample)
    runs = [timed_call(lambda: model.predict(sample)) for _ in range(100)]
    return statistics.median(runs), runs


def peak_memory_mb(function: Callable[[], Any]) -> float:
    """Return peak process RSS observed while the supplied function runs."""
    try:
        peak = memory_usage((function, (), {}), interval=0.01, max_usage=True, include_children=True)
        return float(peak)
    except Exception:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def serialize_model(model: Any, path: Path) -> int:
    joblib.dump(model, path, compress=0)
    return path.stat().st_size


def evaluate_budget(memory_mb: float, latency_ms: float, size_kb: float, tier: str) -> str:
    if tier == "Cloud":
        return "Yes" if latency_ms <= 100 and size_kb <= 500 * 1024 else "No"
    if tier == "Edge":
        return "Yes" if memory_mb <= 1024 and latency_ms <= 50 and size_kb <= 50 * 1024 else "No"
    if tier == "Mobile":
        return "Yes" if memory_mb <= 256 and latency_ms <= 20 and size_kb <= 10 * 1024 else "No"
    if tier == "TinyML":
        return "Yes" if memory_mb * 1024 <= 256 and latency_ms <= 10 and size_kb <= 100 else "No"
    raise ValueError(tier)


def main() -> None:
    set_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)

    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=SEED
    )

    factories = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
    models: dict[str, Any] = {}
    accuracy_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []

    for name, factory in factories.items():
        model = factory().fit(x_train, y_train)
        models[name] = model
        accuracy_rows.append({"model": name, "test_accuracy": f"{accuracy_score(y_test, model.predict(x_test)):.4f}"})

        train_median, train_runs = median_training_time(factory, x_train, y_train)
        sample = x_test[:1]
        inference_median, inference_runs = median_inference_time(model, sample)
        train_peak = peak_memory_mb(lambda: factory().fit(x_train, y_train))
        inference_peak = peak_memory_mb(lambda: model.predict(sample))

        model_path = RESULTS / f"{name}.joblib"
        model_bytes = serialize_model(model, model_path)
        model_kb = model_bytes / 1024.0
        measurement_rows.append(
            {
                "model": name,
                "test_accuracy": f"{accuracy_score(y_test, model.predict(x_test)):.4f}",
                "training_time_median_ms": f"{train_median:.6f}",
                "training_time_runs_ms": json.dumps([round(v, 6) for v in train_runs]),
                "single_sample_inference_median_ms": f"{inference_median:.6f}",
                "single_sample_inference_runs_ms": json.dumps([round(v, 6) for v in inference_runs]),
                "model_size_bytes": model_bytes,
                "model_size_kb": f"{model_kb:.3f}",
                "peak_training_memory_mb": f"{train_peak:.3f}",
                "peak_inference_memory_mb": f"{inference_peak:.3f}",
            }
        )

    with (RESULTS / "baseline_accuracy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "test_accuracy"])
        writer.writeheader()
        writer.writerows(accuracy_rows)

    with (RESULTS / "measurement_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurement_rows[0]))
        writer.writeheader()
        writer.writerows(measurement_rows)

    combined_path = RESULTS / "model.joblib"
    serialize_model(models, combined_path)

    budget_rows: list[dict[str, str]] = []
    for row in measurement_rows:
        memory = max(float(row["peak_training_memory_mb"]), float(row["peak_inference_memory_mb"]))
        latency = float(row["single_sample_inference_median_ms"])
        size_kb = float(row["model_size_kb"])
        for tier in ("Cloud", "Edge", "Mobile", "TinyML"):
            budget_rows.append(
                {
                    "model": row["model"],
                    "deployment_tier": tier,
                    "memory_measured_mb": f"{memory:.3f}",
                    "latency_measured_ms": f"{latency:.6f}",
                    "model_size_measured_kb": f"{size_kb:.3f}",
                    "meets_budget": evaluate_budget(memory, latency, size_kb, tier),
                }
            )
    with (RESULTS / "resource_budget.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(budget_rows[0]))
        writer.writeheader()
        writer.writerows(budget_rows)

    print(pd.DataFrame(measurement_rows).to_string(index=False))
    print(f"Combined model artifact: {combined_path}")


if __name__ == "__main__":
    main()
