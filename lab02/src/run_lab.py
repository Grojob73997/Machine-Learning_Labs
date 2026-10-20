"""Run all required Lab 2 experiments and write evidence to lab02/results."""
from pathlib import Path
import csv
import json
import os
import random
import time

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
os.environ.setdefault("MPLCONFIGDIR", str(LAB.parents[0] / "tmp" / "matplotlib-lab02"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits, threadpool_info

from network import initialize, forward, backward, cross_entropy, evaluate
from gradient_check import check_gradients
from print_versions import version_report

WIDTHS = (16, 64)
RATES = (0.001, 0.01, 0.1)
EPOCHS = 50
BATCH = 32


def save_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    train_indices, test_indices = train_test_split(
        np.arange(len(X)), test_size=0.3, stratify=y, random_state=42
    )
    Y = np.eye(10)[y]
    assert len(train_indices) == 1257 and len(test_indices) == 540
    assert np.all((X >= 0) & (X <= 1))
    assert not set(train_indices).intersection(test_indices)
    counts = [{"digit": k, "train": int(np.sum(y[train_indices] == k)),
               "test": int(np.sum(y[test_indices] == k))} for k in range(10)]
    save_csv("class_counts.csv", counts)
    np.savez(RESULTS / "split_indices.npz", train=train_indices, test=test_indices)
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]


def train(width, rate, X, Y):
    parameters = initialize(width)
    shuffle_rng = np.random.RandomState(42)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = shuffle_rng.permutation(len(X))
        for start in range(0, len(X), BATCH):
            indices = order[start:start + BATCH]
            _, cache = forward(X[indices], parameters)
            gradients = backward(parameters, cache, Y[indices])
            for name in parameters:
                assert gradients[name].shape == parameters[name].shape
                parameters[name] -= rate * gradients[name]
        loss, accuracy = evaluate(parameters, X, Y)
        assert np.isfinite(loss)
        history.append({"width": width, "learning_rate": rate, "epoch": epoch,
                        "train_loss": loss, "train_accuracy": accuracy})
        print(f"W={width:2d} lr={rate:g} epoch={epoch:02d} loss={loss:.6f} accuracy={accuracy:.4f}")
    return parameters, history


def measure_passes(parameters, X, Y):
    _, cache = forward(X, parameters)
    # One warm-up for each operation, followed by 100 separately timed runs.
    forward(X, parameters)
    backward(parameters, cache, Y)
    raw = []
    for repeat in range(1, 101):
        started = time.perf_counter_ns()
        forward(X, parameters)
        forward_ns = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        backward(parameters, cache, Y)
        backward_ns = time.perf_counter_ns() - started
        raw.append({"repeat": repeat, "forward_ns": forward_ns, "backward_ns": backward_ns})
    forward_us = float(np.median([r["forward_ns"] for r in raw]) / 1000)
    backward_us = float(np.median([r["backward_ns"] for r in raw]) / 1000)
    summary = {"batch_size": len(X), "repeats": 100, "forward_median_us": forward_us,
               "backward_median_us": backward_us, "backward_over_forward": backward_us / forward_us}
    return summary, raw


def plot_losses(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, width in zip(axes, WIDTHS):
        for rate in RATES:
            rows = [r for r in history if r["width"] == width and r["learning_rate"] == rate]
            ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows],
                    label=f"lr = {rate}", linewidth=1.8)
        ax.axhline(np.log(10), color="gray", linestyle=":", linewidth=1, label="ln(10)")
        ax.set(title=f"Hidden width = {width}", xlabel="Epoch", xlim=(1, 50))
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Mean training cross-entropy")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    versions = version_report()
    print(versions)
    (RESULTS / "versions.txt").write_text(versions, encoding="utf-8")
    history, summaries, checks, sanity, timing, raw_timing = [], [], [], [], [], []
    with threadpool_limits(limits=1, user_api="blas"):
        X_train, X_test, Y_train, Y_test = prepare_data()
        for width in WIDTHS:
            parameters = initialize(width)
            probabilities, cache = forward(X_train, parameters)
            assert probabilities.shape == (len(X_train), 10)
            row_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
            assert row_error <= 1e-6
            initial_loss = cross_entropy(cache[5], Y_train)
            assert abs(initial_loss - np.log(10)) < 0.01
            sanity.append({"width": width, "initial_loss": initial_loss,
                           "ln10": float(np.log(10)), "max_probability_sum_error": row_error,
                           "parameters": sum(p.size for p in parameters.values())})
            # Fixed small batch, selected before training; no retries or replacements.
            records = check_gradients(parameters, X_train[:8], Y_train[:8])
            checks.extend(dict(width=width, **r) for r in records)
            save_csv("gradient_check.csv", checks)
            passed = sum(r["passed"] for r in records)
            print(f"Gradient check before training, width {width}: {passed}/10", flush=True)
            assert passed >= 9, "Gradient check failed: inspect the saved rows before training."
            for rate in RATES:
                parameters, run_history = train(width, rate, X_train, Y_train)
                history.extend(run_history)
                test_loss, test_accuracy = evaluate(parameters, X_test, Y_test)
                summaries.append(dict(**run_history[-1], test_loss=test_loss, test_accuracy=test_accuracy))
                np.savez(RESULTS / f"model_w{width}_lr{rate:g}.npz", **parameters)
                # Record all six runs: timing is reported independently from training.
                measured, raw = measure_passes(parameters, X_train[:BATCH], Y_train[:BATCH])
                timing.append(dict(width=width, learning_rate=rate, **measured))
                raw_timing.extend(dict(width=width, learning_rate=rate, **r) for r in raw)
        environment = {"arithmetic": "float64 NumPy; Decimal precision 50 for numerical checking only",
                       "seed": 42, "blas_thread_limit": 1, "blas": threadpool_info(),
                       "timing_warmups": 1, "timing_repeats": 100,
                       "initialization_mean": 0.0, "initialization_standard_deviation": 0.01}
    save_csv("sanity_checks.csv", sanity)
    save_csv("epoch_log.csv", history)
    save_csv("summary.csv", summaries)
    save_csv("timing.csv", timing)
    save_csv("timing_raw.csv", raw_timing)
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    plot_losses(history)
    print("All six experiments finished. Evidence saved to", RESULTS)


if __name__ == "__main__":
    main()
