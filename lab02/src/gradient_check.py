"""Central differences using an independent, high-precision forward calculation.

Decimal is from Python's standard library. It avoids cancellation when subtracting
two losses near 2.3026. Training and analytical backpropagation remain float64 NumPy.
"""
from decimal import Decimal, localcontext
import numpy as np
from network import forward, backward


def decimal_loss(X, Y, parameters):
    matrices = {name: [[Decimal.from_float(float(v)) for v in row] for row in parameters[name]]
                for name in ("W1", "W2", "W3")}
    biases = {name: [Decimal.from_float(float(v)) for v in parameters[name]]
              for name in ("b1", "b2", "b3")}
    losses, masks = [], []
    for x, target in zip(X, Y.argmax(axis=1)):
        activations = [Decimal.from_float(float(v)) for v in x]
        for layer in (1, 2, 3):
            W, b = matrices[f"W{layer}"], biases[f"b{layer}"]
            logits = [sum(a * row[j] for a, row in zip(activations, W)) + b[j]
                      for j in range(len(b))]
            if layer < 3:
                masks.extend(v > 0 for v in logits)
                activations = [max(Decimal(0), v) for v in logits]
        largest = max(logits)
        log_sum = sum((v - largest).exp() for v in logits).ln()
        losses.append(log_sum - (logits[int(target)] - largest))
    return sum(losses) / len(losses), masks


def check_gradients(parameters, X, Y):
    _, cache = forward(X, parameters)
    gradients = backward(parameters, cache, Y)
    names = ("W1", "W2", "W3")
    ends = np.cumsum([parameters[name].size for name in names])
    chosen = np.random.RandomState(42).choice(ends[-1], 10, replace=False)
    records = []
    with localcontext() as context:
        context.prec = 50
        for flat in chosen:
            layer = int(np.searchsorted(ends, flat, side="right"))
            name = names[layer]
            offset = int(flat - (ends[layer - 1] if layer else 0))
            index = np.unravel_index(offset, parameters[name].shape)
            original = parameters[name][index]
            try:
                parameters[name][index] = original + 1e-5
                plus, mask_plus = decimal_loss(X, Y, parameters)
                parameters[name][index] = original - 1e-5
                minus, mask_minus = decimal_loss(X, Y, parameters)
            finally:
                parameters[name][index] = original
            numerical = float((plus - minus) / Decimal("0.00002"))
            analytical = float(gradients[name][index])
            relative_error = abs(analytical - numerical) / max(1e-12, abs(analytical) + abs(numerical))
            records.append({
                "parameter": name, "row": int(index[0]), "column": int(index[1]),
                "analytical": analytical, "numerical": numerical,
                "relative_error": relative_error,
                "relu_kink_crossed": mask_plus != mask_minus,
                "passed": relative_error < 1e-6,
            })
    return records
