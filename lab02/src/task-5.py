# G M Robi Farhan Rojob | 73997 | Lab 2
# Run this file to execute tasks 1 through 5.
# Earlier tasks provide the data and functions used below.
from pathlib import Path
from runpy import run_path

previous_tasks = run_path(str(Path(__file__).with_name('task-4.py')))
globals().update({name: value for name, value in previous_tasks.items()
                  if not name.startswith('__')})

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

