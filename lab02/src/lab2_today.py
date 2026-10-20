# TASK 1 - DATASET
from pathlib import Path
from decimal import Decimal, localcontext
import csv
import random
import time
import platform
from importlib import metadata

import numpy as np
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


# TASK 2 - NETWORK DESIGN AND INITIALIZATION
def initialize(width):
    rng = np.random.RandomState(42)
    return {
        "W1": rng.normal(0, 0.01, (64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0, 0.01, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0, 0.01, (width, 10)),
        "b3": np.zeros(10),
    }

models = {width: initialize(width) for width in (16, 64)}
print("\nTASK 2")
for width, parameters in models.items():
    print("Hidden width:", width)
    for name, array in parameters.items():
        print(name, array.shape)


# TASK 3 - FORWARD PROPAGATION
def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = np.maximum(0, Z2)
    Z3 = A2 @ p["W3"] + p["b3"]
    shifted = Z3 - Z3.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    P = exponentials / exponentials.sum(axis=1, keepdims=True)
    return P, (X, Z1, A1, Z2, A2, Z3, P)

print("\nTASK 3")
for width, parameters in models.items():
    P, cache = forward(X_train[:32], parameters)
    assert P.shape == (32, 10)
    np.testing.assert_allclose(P.sum(axis=1), 1, atol=1e-6, rtol=0)
    print("Width", width, "output shape:", P.shape, "row sums: PASS")


# TASK 4 - CROSS-ENTROPY LOSS
def cross_entropy(logits, Y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.sum(Y * log_probabilities) / len(Y))

print("\nTASK 4")
for width, parameters in models.items():
    _, cache = forward(X_train, parameters)
    initial_loss = cross_entropy(cache[5], Y_train)
    assert abs(initial_loss - np.log(10)) < 0.01
    print(f"Width {width}: initial loss = {initial_loss:.8f}")
print("Expected approximately:", np.log(10))


# TASK 5 - BACKPROPAGATION AND NUMERICAL GRADIENT CHECK
def backward(p, cache, Y):
    X, Z1, A1, Z2, A2, Z3, P = cache
    D3 = (P - Y) / len(X)
    D2 = (D3 @ p["W3"].T) * (Z2 > 0)
    D1 = (D2 @ p["W2"].T) * (Z1 > 0)
    return {
        "W1": X.T @ D1, "b1": D1.sum(axis=0),
        "W2": A1.T @ D2, "b2": D2.sum(axis=0),
        "W3": A2.T @ D3, "b3": D3.sum(axis=0),
    }

def reference_loss(X, Y, p):
    # High precision is used ONLY for the independent numerical check.
    weights = {k: [[Decimal.from_float(float(v)) for v in row] for row in p[k]]
               for k in ("W1", "W2", "W3")}
    biases = {k: [Decimal.from_float(float(v)) for v in p[k]]
              for k in ("b1", "b2", "b3")}
    losses = []
    for x, target in zip(X, Y.argmax(axis=1)):
        a = [Decimal.from_float(float(v)) for v in x]
        for layer in (1, 2, 3):
            W, b = weights[f"W{layer}"], biases[f"b{layer}"]
            z = [sum(value * row[j] for value, row in zip(a, W)) + b[j]
                 for j in range(len(b))]
            if layer < 3:
                a = [max(Decimal(0), value) for value in z]
        maximum = max(z)
        losses.append(sum((value - maximum).exp() for value in z).ln()
                      - (z[int(target)] - maximum))
    return sum(losses) / len(losses)

print("\nTASK 5")
gradient_rows = []
for width, p in models.items():
    X_small, Y_small = X_train[:8], Y_train[:8]
    _, cache = forward(X_small, p)
    gradients = backward(p, cache, Y_small)
    names = ("W1", "W2", "W3")
    ends = np.cumsum([p[name].size for name in names])
    chosen = np.random.RandomState(42).choice(ends[-1], 10, replace=False)
    passed = 0
    with localcontext() as context:
        context.prec = 50
        for flat in chosen:
            layer = int(np.searchsorted(ends, flat, side="right"))
            name = names[layer]
            offset = int(flat - (ends[layer - 1] if layer else 0))
            index = np.unravel_index(offset, p[name].shape)
            original = p[name][index]
            try:
                p[name][index] = original + 1e-5
                plus = reference_loss(X_small, Y_small, p)
                p[name][index] = original - 1e-5
                minus = reference_loss(X_small, Y_small, p)
            finally:
                p[name][index] = original
            numerical = float((plus - minus) / Decimal("0.00002"))
            analytical = float(gradients[name][index])
            error = abs(analytical - numerical) / max(1e-12, abs(analytical) + abs(numerical))
            passed += error < 1e-6
            gradient_rows.append(dict(width=width, parameter=name, index=str(index),
                                      analytical=analytical, numerical=numerical,
                                      relative_error=error, passed=error < 1e-6))
            print(f"W={width} {name}{index}: relative error={error:.3e}")
    print(f"Width {width}: {passed}/10 passed")
    assert passed >= 9, "Gradient check failed. Do not continue training."
save_csv("today_gradient_check.csv", gradient_rows)


# TASK 6 - TRAINING AND LOSS CURVES
def evaluate(p, X, Y):
    P, cache = forward(X, p)
    return cross_entropy(cache[5], Y), float(np.mean(P.argmax(axis=1) == Y.argmax(axis=1)))

print("\nTASK 6")
history, summaries, trained_models = [], [], {}
for width in (16, 64):
    for rate in (0.001, 0.01, 0.1):
        p = initialize(width)
        shuffle_rng = np.random.RandomState(42)
        for epoch in range(1, 51):
            order = shuffle_rng.permutation(len(X_train))
            for start in range(0, len(order), 32):
                indices = order[start:start + 32]
                _, cache = forward(X_train[indices], p)
                gradients = backward(p, cache, Y_train[indices])
                for name in p:
                    p[name] -= rate * gradients[name]
            loss, accuracy = evaluate(p, X_train, Y_train)
            history.append(dict(width=width, learning_rate=rate, epoch=epoch,
                                train_loss=loss, train_accuracy=accuracy))
            print(f"W={width} lr={rate} epoch={epoch:02d} loss={loss:.6f}")
        test_loss, test_accuracy = evaluate(p, X_test, Y_test)
        summaries.append(dict(width=width, learning_rate=rate,
                              train_loss=loss, test_loss=test_loss, test_accuracy=test_accuracy))
        trained_models[(width, rate)] = p
save_csv("today_epoch_log.csv", history)
save_csv("today_accuracy.csv", summaries)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, width in zip(axes, (16, 64)):
    for rate in (0.001, 0.01, 0.1):
        rows = [r for r in history if r["width"] == width and r["learning_rate"] == rate]
        ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label=f"lr={rate}")
    ax.set(title=f"Hidden width {width}", xlabel="Epoch", ylabel="Training loss")
    ax.grid(alpha=0.2)
    ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "loss_curves.png", dpi=180)
plt.close(fig)
for row in summaries:
    print(f"Width {row['width']}, lr={row['learning_rate']}: test accuracy {row['test_accuracy']:.2%}")


# TASK 7 - FORWARD/BACKWARD TIMING
print("\nTASK 7")
timing_rows = []
for (width, rate), p in trained_models.items():
    X_batch, Y_batch = X_train[:32], Y_train[:32]
    _, cache = forward(X_batch, p)
    forward(X_batch, p)
    backward(p, cache, Y_batch)
    forward_times, backward_times = [], []
    for _ in range(100):
        start = time.perf_counter_ns()
        forward(X_batch, p)
        forward_times.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        backward(p, cache, Y_batch)
        backward_times.append(time.perf_counter_ns() - start)
    forward_us = float(np.median(forward_times) / 1000)
    backward_us = float(np.median(backward_times) / 1000)
    timing_rows.append(dict(width=width, learning_rate=rate,
                            forward_median_us=forward_us, backward_median_us=backward_us))
    print(f"W={width} lr={rate}: forward={forward_us:.2f} us, backward={backward_us:.2f} us")
save_csv("today_timing.csv", timing_rows)
print("\nCODE COMPLETE. Results folder:", RESULTS)
