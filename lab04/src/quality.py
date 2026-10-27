"""Read-only validation and duplicate/label auditing."""
import numpy as np
import pandas as pd
from schema import FEATURES, NUMERIC, CATEGORIES, RANGES, EXTRA_RANGES, EDUCATION_CODES


def validate(frame):
    """Return all violations and missing counts; never alter or drop rows."""
    expected = FEATURES + ["income"]
    if set(frame.columns) != set(expected):
        raise ValueError(f"Schema columns differ: {list(frame.columns)}")
    violations, summary = [], []

    def record(mask, col, rule):
        for idx, val in frame.loc[mask, col].items():
            violations.append({"row_id": int(idx), "column": col, "rule": rule, "value": str(val)})

    for col in expected:
        s = frame[col]
        marker = s.astype(object).map(lambda x: isinstance(x, str) and x.strip() == "?")
        missing = s.isna() | marker
        summary.append({"column": col, "schema_type": "int" if col in NUMERIC else "category",
                        "storage_dtype": str(s.dtype), "null_count": int(s.isna().sum()),
                        "question_mark_count": int(marker.sum()), "missing_total": int(missing.sum())})
        if col in NUMERIC:
            n = pd.to_numeric(s, errors="coerce")
            bad = (~missing & n.isna()) | (n.notna() & (~np.isfinite(n) | (n % 1 != 0)))
            record(bad, col, "integer_type")
            for rule, ranges in [("manual_range", RANGES), ("dictionary_range", EXTRA_RANGES)]:
                if col in ranges:
                    lo, hi = ranges[col]
                    record(n.notna() & ((n < lo) | ((n > hi) if hi is not None else False)), col, rule)
        else:
            record(~missing & ~s.astype(object).isin(CATEGORIES[col]), col, "unknown_category")
            if col == "income":
                record(missing, col, "missing_target")
    code = frame["education"].astype(object).map(EDUCATION_CODES)
    record(code.notna() & (code != frame["education-num"]), "education-num", "education_code_mismatch")
    return (pd.DataFrame(violations, columns=["row_id", "column", "rule", "value"]),
            pd.DataFrame(summary))


def conflict_audit(frame):
    """Count feature-identical groups with >1 label, treating missing as a value."""
    # Convert categorical columns to objects so groupby never makes unobserved combinations.
    view = frame.astype(object)
    grouped = view.groupby(FEATURES, dropna=False, sort=True, observed=True)["income"]
    counts = grouped.agg(rows="size", label_count="nunique").reset_index()
    conflicts = counts[counts.label_count > 1].copy()
    # Row-aligned transform also handles all-null categorical keys without merge dtype coercion.
    rows = view.loc[grouped.transform("nunique") > 1].reset_index(names="row_id")
    return {"conflicting_feature_groups": len(conflicts), "rows_in_conflicting_groups": len(rows),
            "exact_duplicate_excess_rows": int(frame.duplicated().sum()),
            "feature_duplicate_excess_rows": int(frame.duplicated(FEATURES).sum())}, rows


def overlap_audit(train, test):
    left = train.astype(object).drop_duplicates()
    right = test.astype(object).reset_index(names="row_id")
    exact = right.merge(left, on=FEATURES + ["income"], how="inner")
    features = right.merge(left[FEATURES].drop_duplicates(), on=FEATURES, how="inner")
    return {"test_rows_matching_train_full_row": int(exact.row_id.nunique()),
            "test_rows_matching_train_features": int(features.row_id.nunique())}
