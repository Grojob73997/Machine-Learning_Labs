# G M Robi Farhan Rojob | 73997 | Lab 2
# Run this file to execute tasks 1 through 6.
# Earlier tasks provide the data and functions used below.
from pathlib import Path
from runpy import run_path

previous_tasks = run_path(str(Path(__file__).with_name('task-5.py')))
globals().update({name: value for name, value in previous_tasks.items()
                  if not name.startswith('__')})

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

