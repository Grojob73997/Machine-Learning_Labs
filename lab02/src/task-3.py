# G M Robi Farhan Rojob | 73997 | Lab 2
# Run this file to execute tasks 1 through 3.
# Earlier tasks provide the data and functions used below.
from pathlib import Path
from runpy import run_path

previous_tasks = run_path(str(Path(__file__).with_name('task-2.py')))
globals().update({name: value for name, value in previous_tasks.items()
                  if not name.startswith('__')})

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

