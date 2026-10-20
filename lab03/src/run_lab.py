"""Run the complete Lab 3 experiment. All reported scores are measured."""
from pathlib import Path
import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile

from models import MODELS, EXPECTED_PARAMETERS, hand_costs

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
RESULTS = LAB / "results"


def save_csv(name, rows):
    if rows:
        with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def environment(threads):
    seed_all()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(exist_ok=True, parents=True)
    installed = sorted((d.metadata["Name"], d.version)
                       for d in importlib.metadata.distributions())
    lines = [f"Python {sys.version}", f"Executable: {sys.executable}"]
    lines.extend(f"{name}=={version}" for name, version in installed)
    lines.append("\nCourse requirements status:")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" not in line:
            continue
        name, expected = line.split("==")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "NOT INSTALLED"
        lines.append(f"{name}: required={expected}; installed={actual}")
    (RESULTS / "versions.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    (LAB / "requirements-lock.txt").write_text(
        "\n".join(f"{n}=={v}" for n, v in installed) + "\n", encoding="utf-8")
    record = dict(python=platform.python_version(), platform=platform.platform(),
                  processor=platform.processor(), logical_cpus=os.cpu_count(),
                  torch=torch.__version__, device="cpu", dtype="float32",
                  intraop_threads=threads, interop_threads=1, seed=42,
                  deterministic_algorithms=True, mkldnn=torch.backends.mkldnn.enabled,
                  mha_fastpath_for_latency=torch.backends.mha.get_fastpath_enabled(),
                  train_epochs=5, batch_size=128, learning_rate=0.001,
                  optimizer="Adam", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (RESULTS / "environment.json").write_text(json.dumps(record, indent=2))


def load_data(download=True):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(ROOT / "data", train=True,
                                download=download, transform=transform)
    test = datasets.FashionMNIST(ROOT / "data", train=False,
                                download=download, transform=transform)
    train, val = random_split(full, [55000, 5000],
                             generator=torch.Generator().manual_seed(42))
    assert len(full) == 60000 and len(test) == 10000
    assert set(train.indices).isdisjoint(val.indices)
    np.savez_compressed(RESULTS / "split_indices.npz",
                        train=np.array(train.indices), validation=np.array(val.indices))
    counts = []
    for name, labels in [("train", full.targets[train.indices]),
                         ("validation", full.targets[val.indices]), ("test", test.targets)]:
        for c, n in enumerate(torch.bincount(labels, minlength=10).tolist()):
            counts.append(dict(split=name, label=c, count=n))
    save_csv("class_counts.csv", counts)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "data" / "FashionMNIST" / "raw").glob("*")
              if p.is_file()}
    (RESULTS / "data_sha256.json").write_text(json.dumps(hashes, indent=2))
    print("DATA: 55,000 train / 5,000 validation / 10,000 official test", flush=True)
    return train, val, test


def count_attention_macs(module, inputs, output):
    x = inputs[0]
    b, length, d = x.shape
    # Q/K/V + output projections, QK^T, and attention-weighted values.
    module.total_ops += torch.DoubleTensor([b * (4*length*d*d + 2*length*length*d)])


def count_zero(module, inputs, output):
    # Aligned analytical convention excludes LayerNorm and elementwise work.
    module.total_ops += torch.DoubleTensor([0])


def profile_models():
    costs = hand_costs()
    rows, components = [], []
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    # Fused encoder kernels can bypass Python module hooks in both profilers.
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        for name, constructor in MODELS.items():
            seed_all()
            model = constructor().eval()
            x = torch.zeros(1, 1, 28, 28)
            info = summary(copy.deepcopy(model), input_data=x, verbose=0, depth=5)
            (RESULTS / f"torchinfo_{name}.txt").write_text(str(info), encoding="utf-8")
            actual = sum(p.numel() for p in model.parameters())
            assert actual == EXPECTED_PARAMETERS[name] == info.total_params
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), torch.no_grad():
                native_macs, native_params = profile(copy.deepcopy(model), inputs=(x,),
                                                     verbose=True, report_missing=True)
                aligned_macs, aligned_params = profile(
                    copy.deepcopy(model), inputs=(x,), verbose=True,
                    custom_ops={nn.MultiheadAttention: count_attention_macs,
                                nn.LayerNorm: count_zero})
            (RESULTS / f"thop_{name}.txt").write_text(stream.getvalue(), encoding="utf-8")
            manual = sum(v for _, v in costs[name])
            assert round(2*aligned_macs) == manual, (name, aligned_macs, manual)
            rows.append(dict(model=name, parameters=actual, torchinfo_parameters=info.total_params,
                             manual_flops=manual, native_thop_ops=native_macs,
                             twice_native_thop_ops=2*native_macs,
                             native_thop_parameters=native_params,
                             aligned_thop_macs=aligned_macs, aligned_thop_flops=2*aligned_macs,
                             aligned_thop_parameters=aligned_params,
                             native_gap_flops=manual-2*native_macs))
            components.extend(dict(model=name, component=c, flops=f, macs=f//2)
                              for c, f in costs[name])
            print(f"PROFILE {name}: {actual:,} parameters, {manual:,} hand FLOPs; "
                  f"native THOP={native_macs:,.0f}, aligned THOP={aligned_macs:,.0f} MACs",
                  flush=True)
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
    save_csv("compute_costs.csv", rows)
    save_csv("hand_flop_components.csv", components)
    return rows


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss_total = correct = count = 0
    for x, y in loader:
        logits = model(x)
        loss_total += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return loss_total/count, correct/count, correct, count


@torch.inference_mode()
def latency(model, sample):
    model.eval()
    medians, raw = {}, []
    for batch, runs in [(1, 100), (128, 20)]:
        x = sample[:batch].contiguous()
        model(x)  # One warm-up, excluded.
        measurements = []
        for i in range(runs):
            start = time.perf_counter_ns()
            model(x)
            elapsed = (time.perf_counter_ns()-start)/1e6
            measurements.append(elapsed)
            raw.append(dict(batch_size=batch, run=i+1, milliseconds=elapsed))
        median = statistics.median(measurements)
        medians[f"latency_batch{batch}_ms"] = median
        medians[f"batch{batch}_ms_per_image"] = median/batch
        medians[f"batch{batch}_images_per_second"] = 1000*batch/median
    return medians, raw


def train_models(data, costs):
    train, val, test = data
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)
    sample = next(iter(test_loader))[0]
    summary_rows, epochs, timing_rows = [], [], []
    for name, constructor in MODELS.items():
        seed_all()
        model = constructor()
        train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(42))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, 6):
            model.train()
            start = time.perf_counter()
            loss_sum = correct = total = batches = 0
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()*len(y)
                correct += (logits.detach().argmax(1) == y).sum().item()
                total += len(y)
                batches += 1
            elapsed = time.perf_counter()-start
            vl, va, _, _ = evaluate(model, val_loader)
            epochs.append(dict(model=name, epoch=epoch, train_loss=loss_sum/total,
                               train_accuracy=correct/total, validation_loss=vl,
                               validation_accuracy=va, train_seconds=elapsed,
                               samples=total, optimizer_steps=batches))
            save_csv("epoch_log.csv", epochs)
            print(f"{name} epoch {epoch}/5: loss={loss_sum/total:.4f}, "
                  f"val={va:.4%}, training={elapsed:.1f}s", flush=True)
        test_loss, test_acc, correct_test, n_test = evaluate(model, test_loader)
        torch.save(model.state_dict(), RESULTS / f"model_{name}.pt")
        medians, raw = latency(model, sample)
        timing_rows.extend(dict(model=name, **r) for r in raw)
        row = next(r for r in costs if r["model"] == name).copy()
        row.update(test_loss=test_loss, test_accuracy=test_acc,
                   correct_test=correct_test, test_samples=n_test,
                   final_validation_accuracy=epochs[-1]["validation_accuracy"], **medians)
        summary_rows.append(row)
        save_csv("summary.csv", summary_rows)
        save_csv("latency_raw.csv", timing_rows)
        print(f"FINISHED {name}: test={test_acc:.4%}; B1={medians['latency_batch1_ms']:.4f}ms; "
              f"B128={medians['latency_batch128_ms']:.4f}ms", flush=True)
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    environment(args.threads)
    if args.download_only:
        load_data()
        return
    costs = profile_models()
    if args.profile_only:
        return
    train_models(load_data(), costs)
    from write_report import build_report
    build_report()
    print("COMPLETE: three models, five epochs each, 360 timed runs.", flush=True)


if __name__ == "__main__":
    main()
