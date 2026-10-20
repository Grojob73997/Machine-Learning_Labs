# G M Robi Farhan Rojob | 73997 | Lab 2
# Run this file to execute tasks 1 through 2.
# Earlier tasks provide the data and functions used below.
from pathlib import Path
from runpy import run_path

previous_tasks = run_path(str(Path(__file__).with_name('task-1.py')))
globals().update({name: value for name, value in previous_tasks.items()
                  if not name.startswith('__')})

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


# Draw both network designs using Python (approved for this task).
from matplotlib.patches import Circle


def draw_network(ax, width):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.1)
    ax.set_aspect("equal")
    ax.axis("off")
    parameter_count = sum(array.size for array in models[width].values())
    ax.text(0.2, 5.75, f"64 → {width} → {width} → 10", fontsize=18,
            weight="bold", color="#16324f")
    ax.text(13.8, 5.75, f"{parameter_count:,} trainable parameters", fontsize=11,
            ha="right", color="#45556a")

    positions = [1, 5, 9, 13]
    node_y = [4.1, 3.55, 3.0, 1.9]
    sizes = [64, width, width, 10]
    titles = ["Input", "Hidden layer 1", "Hidden layer 2", "Output"]
    details = ["64 pixels (8 × 8 image)", f"{width} neurons • ReLU",
               f"{width} neurons • ReLU", "10 classes • softmax"]
    colors = ["#2176ae", "#278567", "#278567", "#b86b28"]

    # A few representative neurons are drawn; every adjacent layer is dense.
    for left, right in zip(positions[:-1], positions[1:]):
        for start_y in node_y:
            for end_y in node_y:
                ax.plot([left, right], [start_y, end_y], color="#ccd6e0",
                        linewidth=0.7, zorder=1)

    for layer, x in enumerate(positions):
        ax.text(x, 5.0, titles[layer], ha="center", fontsize=12, weight="bold")
        ax.text(x, 4.65, details[layer], ha="center", fontsize=10, color="#45556a")
        labels = ["1", "2", "3", str(sizes[layer])]
        if layer == 3:
            labels = ["0", "1", "2", "9"]
        for y_position, label in zip(node_y, labels):
            ax.add_patch(Circle((x, y_position), 0.19, facecolor="white",
                                edgecolor=colors[layer], linewidth=1.8, zorder=3))
            ax.text(x, y_position, label, ha="center", va="center", fontsize=9,
                    color=colors[layer], zorder=4)
        ax.text(x, 2.45, "⋮", ha="center", va="center", fontsize=23,
                color=colors[layer], backgroundcolor="white", zorder=4)

    shapes = [f"W1: (64, {width})\nb1: ({width},)",
              f"W2: ({width}, {width})\nb2: ({width},)",
              f"W3: ({width}, 10)\nb3: (10,)"]
    for x, label in zip([3, 7, 11], shapes):
        ax.text(x, 1.25, label, ha="center", va="top", fontsize=11,
                linespacing=1.6, color="#16324f")
    ax.text(7, 0.25,
            f"Batch shapes: (B, 64) → (B, {width}) → (B, {width}) → (B, 10)",
            ha="center", fontsize=11, color="#45556a")


figure, axes = plt.subplots(2, 1, figsize=(14, 12))
figure.suptitle("Lab 2 · Two-hidden-layer neural networks", fontsize=21,
               weight="bold", color="#16324f", y=0.98)
for axis, width in zip(axes, (16, 64)):
    draw_network(axis, width)
figure.text(0.5, 0.055,
            "Dense connections; dots indicate omitted neurons. B = batch size.\n"
            "Weights: Gaussian, mean 0 and standard deviation 0.01. Biases: zero.",
            ha="center", fontsize=11, linespacing=1.6, color="#45556a")
figure.text(0.5, 0.015, "G M Robi Farhan Rojob · 73997 · Group 01",
            ha="center", fontsize=10, color="#45556a")
figure.subplots_adjust(top=0.93, bottom=0.11, left=0.04, right=0.96, hspace=0.1)
figure.savefig(RESULTS / "task-2-network-diagram.png", dpi=180)
figure.savefig(RESULTS / "task-2-network-diagram.pdf")
plt.close(figure)
print("Network diagram saved:", RESULTS / "task-2-network-diagram.png")
print("Printable diagram saved:", RESULTS / "task-2-network-diagram.pdf")

