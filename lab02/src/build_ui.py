"""Run all seven tasks and export their measured results for the HTML interface."""
from datetime import datetime
from pathlib import Path
import csv
import json
import platform
from runpy import run_path

LAB = Path(__file__).resolve().parent.parent
state = run_path(str(Path(__file__).with_name("task-7.py")))


def read_csv(name):
    with (LAB / "results" / name).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            if value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


np = state["np"]
samples = []
for digit in range(10):
    index = int(np.flatnonzero(state["y_test"] == digit)[0])
    samples.append({"label": digit, "testIndex": index,
                    "pixels": state["X_test"][index].tolist()})

models = {}
for (width, rate), parameters in state["trained_models"].items():
    key = f"{width}-{rate}"
    sample_x = np.array([sample["pixels"] for sample in samples])
    probabilities, _ = state["forward"](sample_x, parameters)
    models[key] = {"parameters": {name: value.tolist() for name, value in parameters.items()},
                   "referenceProbabilities": probabilities.tolist()}

initial_losses = {}
for width in (16, 64):
    _, cache = state["forward"](state["X_train"], state["initialize"](width))
    initial_losses[str(width)] = state["cross_entropy"](cache[5], state["Y_train"])

data = {
    "student": "G M Robi Farhan Rojob", "studentId": "73997",
    "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
    "python": platform.python_version(), "seed": 42,
    "trainSize": len(state["X_train"]), "testSize": len(state["X_test"]),
    "trainCounts": np.bincount(state["y_train"], minlength=10).tolist(),
    "testCounts": np.bincount(state["y_test"], minlength=10).tolist(),
    "samples": samples, "models": models, "initialLosses": initial_losses,
    "accuracy": read_csv("today_accuracy.csv"),
    "history": read_csv("today_epoch_log.csv"),
    "gradients": read_csv("today_gradient_check.csv"),
    "timing": read_csv("today_timing.csv"),
}
ui = LAB / "ui"
ui.mkdir(exist_ok=True)
(ui / "data.js").write_text("window.LAB_DATA = " + json.dumps(data, separators=(",", ":"),
                            allow_nan=False) + ";\n", encoding="utf-8")
print("Interface data refreshed:", ui / "index.html")

