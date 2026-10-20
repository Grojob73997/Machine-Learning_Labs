"""Two-hidden-layer network. All model calculations use NumPy."""
import numpy as np


def initialize(width):
    np.random.seed(42)
    return {
        "W1": np.random.normal(0, 0.01, (64, width)),
        "b1": np.zeros(width),
        "W2": np.random.normal(0, 0.01, (width, width)),
        "b2": np.zeros(width),
        "W3": np.random.normal(0, 0.01, (width, 10)),
        "b3": np.zeros(10),
    }


def forward(X, parameters):
    Z1 = X @ parameters["W1"] + parameters["b1"]
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ parameters["W2"] + parameters["b2"]
    A2 = np.maximum(0, Z2)
    Z3 = A2 @ parameters["W3"] + parameters["b3"]
    shifted = Z3 - Z3.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    P = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return P, (X, Z1, A1, Z2, A2, Z3, P)


def cross_entropy(logits, Y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.sum(Y * log_probabilities) / len(Y))


def backward(parameters, cache, Y):
    X, Z1, A1, Z2, A2, Z3, P = cache
    D3 = (P - Y) / len(X)
    dW3 = A2.T @ D3
    db3 = D3.sum(axis=0)
    D2 = (D3 @ parameters["W3"].T) * (Z2 > 0)
    dW2 = A1.T @ D2
    db2 = D2.sum(axis=0)
    D1 = (D2 @ parameters["W2"].T) * (Z1 > 0)
    dW1 = X.T @ D1
    db1 = D1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def evaluate(parameters, X, Y):
    P, cache = forward(X, parameters)
    predictions = P.argmax(axis=1)
    return cross_entropy(cache[5], Y), float(np.mean(predictions == Y.argmax(axis=1)))
