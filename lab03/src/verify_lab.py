"""Independent checks of architecture, patch ordering, gradients and saved evidence."""
from pathlib import Path
import csv
import json
import sys
import torch
from torch import nn
from models import MODELS, EXPECTED_PARAMETERS, Attention

LAB = Path(__file__).resolve().parents[1]
R = LAB / "results"


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for name, constructor in MODELS.items():
        m = constructor()
        assert sum(p.numel() for p in m.parameters()) == EXPECTED_PARAMETERS[name]
        x = torch.randn(3, 1, 28, 28)
        y = m(x)
        assert y.shape == (3, 10) and torch.isfinite(y).all()
        nn.functional.cross_entropy(y, torch.tensor([0, 4, 9])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        print(name, "parameter count, output shape, and finite backward: PASS")
    grid = torch.arange(784).reshape(1, 1, 28, 28).float()
    patches = Attention.patches(grid)
    for r in range(4):
        for c in range(4):
            torch.testing.assert_close(patches[0, 4*r+c], grid[0, 0, r*7:r*7+7, c*7:c*7+7].flatten())
    print("All 16 patches match direct image slices: PASS")
    m = Attention().eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.inference_mode():
        z = m.projection(m.patches(x))
        perm = torch.randperm(16)
        a, b = z, z[:, perm]
        for layer in m.layers:
            a, b = layer(a), layer(b)
        torch.testing.assert_close(m.classifier(a.mean(1)), m.classifier(b.mean(1)), atol=1e-6, rtol=1e-5)
    print("No-position attention has expected patch permutation invariance: PASS")
    if "--results" in sys.argv:
        def read(name):
            with (R / name).open(encoding="utf-8") as f:
                return list(csv.DictReader(f))
        summary, epochs, times = read("summary.csv"), read("epoch_log.csv"), read("latency_raw.csv")
        assert len(summary) == 3 and len(epochs) == 15 and len(times) == 360
        import statistics
        from run_lab import load_data, evaluate
        from torch.utils.data import DataLoader
        _, _, test = load_data(download=False)
        loader = DataLoader(test, batch_size=128)
        for r in summary:
            name = r["model"]
            e = [x for x in epochs if x["model"] == name]
            assert [int(x["epoch"]) for x in e] == [1, 2, 3, 4, 5]
            assert all(int(x["samples"]) == 55000 and int(x["optimizer_steps"]) == 430 for x in e)
            for b, n in [(1,100), (128,20)]:
                t = [float(x["milliseconds"]) for x in times
                     if x["model"] == name and int(x["batch_size"]) == b]
                assert len(t) == n
                assert abs(statistics.median(t)-float(r[f"latency_batch{b}_ms"])) < 1e-12
            m = MODELS[name]()
            m.load_state_dict(torch.load(R / f"model_{name}.pt", map_location="cpu", weights_only=True))
            _, acc, correct, count = evaluate(m, loader)
            assert count == 10000 and correct == int(r["correct_test"])
            assert acc == float(r["test_accuracy"])
            print(name, "checkpoint reproduces reported test accuracy; epochs and latency medians: PASS")
        print("All 15 epochs, 360 timings, and 3 complete test evaluations verified.")


if __name__ == "__main__":
    main()
