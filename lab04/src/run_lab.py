"""Execute the complete Adult pipeline, save evidence, and verify a second fit."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import subprocess
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from quality import validate, conflict_audit, overlap_audit
from schema import FEATURES, NUMERIC, CATEGORICAL, CATEGORIES, RANGES, EXTRA_RANGES
from transforms import build_preprocessor

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def csv_save(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g", na_rep="?")


def frame_digest(frame):
    return hashlib.sha256(frame.to_csv(index=True, lineterminator="\n", na_rep="?",
                                       float_format="%.17g").encode("utf-8")).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment(out, config):
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    text = [f"Python {sys.version}", f"Executable: {sys.executable}"] + [f"{n}=={v}" for n, v in packages]
    (out / "versions.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)
    json_save(out / "environment.json", dict(python=platform.python_version(),
        platform=platform.platform(), seed=config["seed"], blas_threads=1,
        packages=dict(packages), executable=sys.executable))
    (LAB / "requirements-lock.txt").write_text("\n".join(f"{n}=={v}" for n,v in packages) + "\n")


def load_raw(config):
    data = fetch_openml(name=config["dataset"]["name"], version=config["dataset"]["version"],
                       as_frame=True, data_home=str(ROOT / "data" / "openml"))
    if str(data.details["id"]) != config["dataset"]["expected_openml_id"]:
        raise ValueError("Unexpected OpenML dataset ID")
    raw = data.frame
    if raw.shape != (config["dataset"]["expected_rows"], config["dataset"]["expected_columns"]):
        raise ValueError(f"Unexpected raw shape: {raw.shape}")
    return raw, data.details


def split_and_deduplicate(frame, config):
    train_ids, test_ids = train_test_split(frame.index.to_numpy(), test_size=config["test_size"],
        stratify=frame["income"], random_state=config["seed"])
    train_raw, test_raw = frame.loc[train_ids].copy(), frame.loc[test_ids].copy()
    duplicates = train_raw.duplicated(keep="first")
    train = train_raw.loc[~duplicates].copy()
    return train_raw, train, test_raw, train_raw.loc[duplicates].copy()


def fit_variant(train, test, config, engineered):
    pre = build_preprocessor(config, engineered)
    X_train = pre.fit_transform(train[FEATURES])
    X_test = pre.transform(test[FEATURES])
    y_train = (train.income == config["dataset"]["positive_label"]).astype(int).to_numpy()
    y_test = (test.income == config["dataset"]["positive_label"]).astype(int).to_numpy()
    model = LogisticRegression(solver=config["model"]["solver"], C=config["model"]["C"],
        max_iter=config["model"]["max_iter"], tol=config["model"]["tol"], random_state=config["seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= config["model"]["threshold"]).astype(int)
    train_pred = model.predict(X_train)
    metrics = {"variant": "engineered" if engineered else "baseline", "input_features": 16 if engineered else 14,
        "encoded_features": X_train.shape[1], "train_rows": len(train), "test_rows": len(test),
        "accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred, zero_division=0),
        "train_accuracy": accuracy_score(y_train, train_pred), "train_f1": f1_score(y_train, train_pred),
        "iterations": int(model.n_iter_[0]), "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_test, pred).ravel().tolist()}
    return dict(pre=pre, model=model, train=X_train, test=X_test, y_train=y_train,
                y_test=y_test, probability=prob, prediction=pred, metrics=metrics)


def write_sample(frame, out, config, violations):
    sample = frame.sample(n=config["audit_sample_size"], random_state=config["seed"]).copy()
    sample.insert(0, "row_id", sample.index)
    issues = violations.groupby("row_id").apply(
        lambda x: "; ".join(x.column + ":" + x.rule), include_groups=False).to_dict() if len(violations) else {}
    sample["automated_dictionary_check"] = [issues.get(i, "PASS") for i in sample.index]
    csv_save(out / "audit_sample_50.csv", sample)
    worksheet = out / "manual_review_50.csv"
    # This is a human-owned worksheet; reruns must not erase completed annotations.
    if worksheet.exists():
        old = pd.read_csv(worksheet)
        if old.row_id.tolist() != sample.row_id.tolist():
            raise ValueError("Existing manual review IDs differ; preserve the worksheet before a new sampling run")
    else:
        sample["manual_result"] = ""
        sample["manual_notes"] = ""
        sample["reviewer"] = ""
        sample["review_date"] = ""
        csv_save(worksheet, sample)


def run(config, out, replay=True):
    out.mkdir(parents=True, exist_ok=True)
    data_out = ROOT / "data" / "lab04"
    data_out.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    environment(out, config)
    raw, details = load_raw(config)
    raw_hash = frame_digest(raw)
    print(f"INGESTION: {len(raw)} rows, {raw.shape[1]} columns; target stored as class", flush=True)
    raw_file = data_out / "adult_openml_v2_raw.csv"
    if raw_file.exists():
        current_bytes = raw.to_csv(index=False, lineterminator="\n", na_rep="?", float_format="%.17g").encode()
        if hashlib.sha256(current_bytes).hexdigest() != sha(raw_file):
            raise ValueError("Existing immutable raw CSV differs from current OpenML data")
    else:
        csv_save(raw_file, raw)
    json_save(out / "ingestion.json", dict(rows=len(raw), columns=raw.shape[1],
        raw_columns=raw.columns.tolist(), raw_sha256=sha(raw_file), openml_details=details,
        mapping="class -> income on a separate validation/model view; original frame unchanged"))
    frame = raw.rename(columns={config["dataset"]["raw_target"]: "income"}).copy(deep=True)
    violations, missing = validate(frame)
    csv_save(out / "validation_violations.csv", violations)
    csv_save(out / "missing_values.csv", missing)
    json_save(out / "schema.json", dict(types={c:"int" if c in NUMERIC else "category" for c in frame},
        ranges=RANGES, extra_dictionary_ranges=EXTRA_RANGES, allowed_categories=CATEGORIES,
        null_policy="? and parser NaN are allowed feature missing values, never a missing target"))
    if len(violations):
        raise ValueError(f"{len(violations)} validation violations; inspect report CSV, no rows silently removed")
    train_raw, train, test, dropped = split_and_deduplicate(frame, config)
    test_hash = frame_digest(test)
    split = pd.DataFrame({"row_id":frame.index, "split":"train", "kept_for_training":False})
    split.loc[split.row_id.isin(test.index), "split"] = "test"
    split.loc[split.row_id.isin(train.index), "kept_for_training"] = True
    csv_save(out / "split_indices.csv", split)
    csv_save(out / "train_duplicates_removed.csv", dropped.reset_index(names="row_id"))
    counts = []
    for name, rows in [("raw",frame), ("train_before_dedup",train_raw), ("train_after_dedup",train), ("test",test)]:
        for label, n in rows.income.value_counts().sort_index().items():
            counts.append(dict(split=name, income=label, count=int(n), proportion=float(n/len(rows))))
    csv_save(out / "class_counts.csv", pd.DataFrame(counts))
    print(f"SPLIT: {len(train_raw)} train -> {len(train)} after dedup; {len(test)} unchanged test rows", flush=True)
    models = {}
    for engineered in (False, True):
        result = fit_variant(train, test, config, engineered)
        name = result["metrics"]["variant"]
        models[name] = result
        joblib.dump(result["pre"], out / ("preprocessor.joblib" if engineered else "preprocessor_baseline.joblib"))
        joblib.dump(result["model"], out / f"logistic_{name}.joblib")
        csv_save(out / f"predictions_{name}.csv", pd.DataFrame({"row_id":test.index,
            "income_gt_50k":result["y_test"], "predicted":result["prediction"],
            "probability_gt_50k":result["probability"]}))
        print(f"{name}: accuracy={result['metrics']['accuracy']:.6f}, F1={result['metrics']['f1']:.6f}", flush=True)
    metrics = [m["metrics"] for m in models.values()]
    json_save(out / "metrics.json", metrics)
    csv_save(out / "metrics.csv", pd.DataFrame([{k:v for k,v in m.items() if not isinstance(v,list)} for m in metrics]))
    final = models["engineered"]
    names = final["pre"].named_steps["encode"].get_feature_names_out().tolist()
    json_save(out / "feature_names.json", names)
    parts = []
    for name, rows, arr, labels in [("train",train,final["train"],final["y_train"]),
                                    ("test",test,final["test"],final["y_test"])]:
        table = pd.DataFrame(arr, columns=names)
        table.insert(0, "income_gt_50k", labels)
        table.insert(0, "row_id", rows.index.to_numpy())
        table.insert(0, "split", name)
        parts.append(table)
    processed_path = data_out / "processed_dataset.csv"
    csv_save(processed_path, pd.concat(parts, ignore_index=True))
    processed_hash = sha(processed_path)
    clean = final["pre"].named_steps["clean"]
    params = {"numeric_columns":NUMERIC, "categorical_columns":CATEGORICAL,
        "numeric_medians":clean.numeric_imputer_.statistics_.tolist(),
        "categorical_modes":clean.categorical_imputer_.statistics_.tolist(),
        "clip_lower":clean.lower_.tolist(), "clip_upper":clean.upper_.tolist(),
        "fit_row_count":len(clean.fit_row_ids_)}
    json_save(out / "fitted_parameters.json", params)
    unclipped = train[NUMERIC].astype(float).fillna(pd.Series(params["numeric_medians"],index=NUMERIC))
    clip_counts = []
    for i,c in enumerate(NUMERIC):
        clip_counts.append(dict(column=c, lower=float(clean.lower_[i]), upper=float(clean.upper_[i]),
            train_values_clipped=int(((unclipped[c]<clean.lower_[i])|(unclipped[c]>clean.upper_[i])).sum())))
    csv_save(out / "clipping.csv", pd.DataFrame(clip_counts))
    # Audits are reporting only. Their outcomes never select or fit a model.
    conflicts, conflict_rows = conflict_audit(frame)
    overlap = overlap_audit(train, test)
    csv_save(out / "conflicting_labels.csv", conflict_rows)
    write_sample(frame, out, config, violations)
    json_save(out / "quality_audit.json", dict(**conflicts, **overlap,
        sample_size=config["audit_sample_size"], sample_impossible_values=0,
        manual_review_status="Pending student review: results/manual_review_50.csv",
        label_limit="Dictionary and duplicate consistency cannot verify actual personal income"))
    proof = dict(split_disjoint=bool(set(train_raw.index).isdisjoint(test.index)),
        fit_ids_are_kept_train=bool(np.array_equal(clean.fit_row_ids_,train.index.to_numpy())),
        raw_unchanged=(raw_hash==frame_digest(raw)), test_frame_unchanged=(test_hash==frame_digest(test)),
        no_duplicate_full_train_rows=(not bool(train.duplicated().any())),
        test_row_count_preserved=(len(test)==len(split[split.split=="test"])),
        reload_transform_identical=False, replay_performed=replay)
    reload_pre = joblib.load(out / "preprocessor.joblib")
    proof["reload_transform_identical"] = bool(np.array_equal(reload_pre.transform(test[FEATURES]), final["test"]))
    if replay:
        seed_all(config["seed"])
        _, replay_train, replay_test, _ = split_and_deduplicate(frame, config)
        equal = []
        for engineered in (False,True):
            replay_result = fit_variant(replay_train, replay_test, config, engineered)
            before = models[replay_result["metrics"]["variant"]]
            equal.append(all(np.array_equal(before[k],replay_result[k]) for k in
                             ["train","test","probability","prediction"]))
        proof["second_fit_arrays_and_predictions_identical"] = all(equal)
        proof["replay_split_ids_identical"] = bool(np.array_equal(train.index,replay_train.index) and
                                                   np.array_equal(test.index,replay_test.index))
        replay_final = replay_result
        replay_parts = []
        for split_name, rows, arr, labels in [("train",replay_train,replay_final["train"],replay_final["y_train"]),
                                             ("test",replay_test,replay_final["test"],replay_final["y_test"])]:
            table = pd.DataFrame(arr,columns=names)
            table.insert(0,"income_gt_50k",labels)
            table.insert(0,"row_id",rows.index.to_numpy())
            table.insert(0,"split",split_name)
            replay_parts.append(table)
        replay_hash = hashlib.sha256(pd.concat(replay_parts,ignore_index=True).to_csv(index=False,
            lineterminator="\n",float_format="%.17g",na_rep="?").encode()).hexdigest()
        proof["processed_sha256_matches_second_fit"] = replay_hash == processed_hash
    # Every reported verification must be true; replay_performed may be disabled explicitly.
    required = {k:v for k,v in proof.items() if k != "replay_performed"}
    if not all(required.values()):
        raise AssertionError(proof)
    json_save(out / "reproducibility.json", proof)
    run_config = dict(config)
    run_config["artifacts"] = dict(processed_dataset="data/lab04/processed_dataset.csv",
        sha256=processed_hash, raw_dataset="data/lab04/adult_openml_v2_raw.csv", raw_sha256=sha(raw_file),
        serialization="CSV, UTF-8, LF, header, no index, float %.17g, train rows then test rows; source row_id and labels included",
        transformed_columns=len(names), processed_rows=len(train)+len(test))
    (out / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (out / "processed_dataset.sha256").write_text(processed_hash + "  data/lab04/processed_dataset.csv\n")
    for col in ("sex", "race"):
        csv_save(out / f"composition_{col}.csv", frame.groupby(col,observed=True).agg(
            rows=("income","size"), positive_labels=("income",lambda x:int((x==">50K").sum()))).reset_index())
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(2)
    ax.bar(x-.18,[m["accuracy"] for m in metrics],.36,label="Accuracy")
    ax.bar(x+.18,[m["f1"] for m in metrics],.36,label="F1 (>50K)")
    ax.set(xticks=x,xticklabels=["14 raw features","+ 2 derived features"],ylim=(0,1),
           ylabel="Held-out score",title="Adult: fixed 70/30 stratified split (seed 42)")
    ax.legend(loc="lower right")
    for bars in ax.containers:
        ax.bar_label(bars,fmt="%.4f",padding=3)
    fig.tight_layout()
    fig.savefig(out / "feature_comparison.png",dpi=170)
    plt.close(fig)
    from write_report import write_documents
    write_documents(LAB, out, config, metrics, missing, conflicts, overlap, len(train_raw), len(train), len(test), processed_hash)
    print("PIPELINE COMPLETE; human 50-row review remains in manual_review_50.csv", flush=True)


class Tee:
    def __init__(self,*streams): self.streams = streams
    def write(self,s):
        for stream in self.streams: stream.write(s)
        return len(s)
    def flush(self):
        for stream in self.streams: stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=LAB / "config.yaml")
    parser.add_argument("--results",type=Path,default=LAB / "results")
    parser.add_argument("--skip-replay",action="store_true",help="Skip independent second fit (default verifies it)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # These explanatory config fields describe the fixed algorithms implemented by this lab.
    supported = {"numeric_imputer":"median", "categorical_imputer":"most_frequent",
        "duplicate_policy":"exact_full_rows_train_only_keep_first_before_imputation",
        "numeric_scaling":"StandardScaler", "categorical_encoding":"OneHotEncoder_handle_unknown_ignore",
        "derived_order":"after_imputation_and_clipping_before_scaling",
        "derived_features":{"capital_net":"capital-gain - capital-loss",
                            "hours_per_age":"hours-per-week / (age - 16)"}}
    for key,value in supported.items():
        if config["preprocessing"][key] != value:
            raise ValueError(f"Unsupported preprocessing setting: {key}")
    if config["seed"] != 42 or config["test_size"] != .30 or config["audit_sample_size"] != 50:
        raise ValueError("This lab requires seed 42, test_size 0.30, and 50 audit rows")
    args.results.mkdir(parents=True,exist_ok=True)
    with (args.results / "run_log.txt").open("w",encoding="utf-8") as log, \
         contextlib.redirect_stdout(Tee(sys.stdout,log)), threadpool_limits(limits=1):
        run(config,args.results,replay=not args.skip_replay)
        import unittest
        suite = unittest.defaultTestLoader.discover(str(LAB / "src"),pattern="test_*.py")
        with (args.results / "verification.txt").open("w",encoding="utf-8") as verification:
            result = unittest.TextTestRunner(stream=Tee(sys.stdout,verification),verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        check = subprocess.run([sys.executable,str(LAB / "src" / "verify_lab.py"),str(args.results.resolve())],
            capture_output=True,text=True)
        print(check.stdout,end="")
        with (args.results / "verification.txt").open("a",encoding="utf-8") as verification:
            verification.write(check.stdout+check.stderr)
        if check.returncode:
            raise RuntimeError(check.stderr)
        print("COMPLETE: pipeline, reports, serialized models, replay, and verification passed.",flush=True)


if __name__ == "__main__":
    main()
