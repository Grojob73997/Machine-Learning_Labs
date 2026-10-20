# G M Robi Farhan Rojob | 73997 | Lab 2
# Run this file to execute tasks 1 through 4.
# Earlier tasks provide the data and functions used below.
from pathlib import Path
from runpy import run_path

previous_tasks = run_path(str(Path(__file__).with_name('task-3.py')))
globals().update({name: value for name, value in previous_tasks.items()
                  if not name.startswith('__')})

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

