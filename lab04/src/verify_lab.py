"""Check saved evidence in a fresh Python process without retraining."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from schema import FEATURES, NUMERIC, CATEGORICAL
from transforms import normalize_features
from run_lab import sha, load_raw, split_and_deduplicate, LAB, ROOT


def verify(out):
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    raw,_ = load_raw(cfg)
    frame = raw.rename(columns={"class":"income"})
    train_raw,train,test,_ = split_and_deduplicate(frame,cfg)
    split = pd.read_csv(out / "split_indices.csv")
    assert len(split)==48842 and split.row_id.nunique()==48842
    assert set(split.loc[split.split=="test","row_id"])==set(test.index)
    assert set(split.loc[split.kept_for_training,"row_id"])==set(train.index)
    assert set(train_raw.index).isdisjoint(test.index)
    assert len(test)==14653 and len(train_raw)==34189 and len(train)==34165
    metrics = json.loads((out / "metrics.json").read_text())
    for variant in ("baseline","engineered"):
        pre = joblib.load(out / ("preprocessor.joblib" if variant=="engineered" else "preprocessor_baseline.joblib"))
        model = joblib.load(out / f"logistic_{variant}.joblib")
        clean = pre.named_steps["clean"]
        train_norm = normalize_features(train[FEATURES])
        expected_medians = train_norm[NUMERIC].median().to_numpy()
        np.testing.assert_array_equal(clean.numeric_imputer_.statistics_,expected_medians)
        expected_modes = [train_norm[c].mode().iloc[0] for c in CATEGORICAL]
        np.testing.assert_array_equal(clean.categorical_imputer_.statistics_,expected_modes)
        imputed = train_norm[NUMERIC].fillna(dict(zip(NUMERIC,expected_medians)))
        np.testing.assert_allclose(clean.lower_,imputed.quantile(.01).to_numpy(),rtol=0,atol=1e-10)
        np.testing.assert_allclose(clean.upper_,imputed.quantile(.99).to_numpy(),rtol=0,atol=1e-10)
        np.testing.assert_array_equal(clean.fit_row_ids_,train.index.to_numpy())
        X = pre.transform(test[FEATURES])
        probability = model.predict_proba(X)[:,1]
        pred = (probability>=.5).astype(int)
        saved = pd.read_csv(out / f"predictions_{variant}.csv",float_precision="round_trip")
        np.testing.assert_array_equal(saved.row_id,test.index)
        np.testing.assert_array_equal(saved.predicted,pred)
        np.testing.assert_array_equal(saved.probability_gt_50k,probability)
        y = (test.income==">50K").astype(int)
        m = next(m for m in metrics if m["variant"]==variant)
        assert m["accuracy"]==accuracy_score(y,pred) and m["f1"]==f1_score(y,pred)
        if variant=="engineered":
            table = pd.read_csv(ROOT / cfg["artifacts"]["processed_dataset"],float_precision="round_trip")
            names = pre.named_steps["encode"].get_feature_names_out().tolist()
            np.testing.assert_array_equal(table.loc[table.split=="test",names].to_numpy(),X)
            np.testing.assert_array_equal(table.loc[table.split=="train",names].to_numpy(),pre.transform(train[FEATURES]))
            assert sha(ROOT / cfg["artifacts"]["processed_dataset"])==cfg["artifacts"]["sha256"]
    assert sha(ROOT / cfg["artifacts"]["raw_dataset"])==cfg["artifacts"]["raw_sha256"]
    print("PASS: independent train medians, modes and percentiles; source/split IDs; fresh-process joblib reload; "
          "both saved prediction arrays and scores; processed train/test matrices; raw and processed SHA256.")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv)>1 else LAB / "results"
    with threadpool_limits(limits=1):
        verify(out)
