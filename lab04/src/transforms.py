"""Importable sklearn transformers; joblib can reload them in a fresh process."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from schema import NUMERIC, CATEGORICAL, FEATURES


def normalize_features(X):
    """Stateless normalization on a copy; '?' and parser NaN mean missing."""
    out = X.loc[:, FEATURES].copy()
    for col in CATEGORICAL:
        out[col] = out[col].astype(object).map(
            lambda x: np.nan if pd.isna(x) or str(x).strip() == "?" else str(x).strip())
    for col in NUMERIC:
        values = out[col].astype(object).map(
            lambda x: np.nan if pd.isna(x) or (isinstance(x, str) and x.strip() == "?") else x)
        out[col] = pd.to_numeric(values, errors="raise").astype(float)
    return out


class AdultCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        data = normalize_features(X)
        self.fit_row_ids_ = X.index.to_numpy(copy=True)
        self.numeric_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
        numbers = self.numeric_imputer_.fit_transform(data[NUMERIC])
        self.categorical_imputer_.fit(data[CATEGORICAL])
        self.lower_ = np.quantile(numbers, self.lower, axis=0, method="linear")
        self.upper_ = np.quantile(numbers, self.upper, axis=0, method="linear")
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        out = normalize_features(X)
        out[NUMERIC] = np.clip(self.numeric_imputer_.transform(out[NUMERIC]), self.lower_, self.upper_)
        out[CATEGORICAL] = self.categorical_imputer_.transform(out[CATEGORICAL])
        return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        if self.enabled:
            denominator = out["age"] - 16
            if (denominator <= 0).any():
                raise ValueError("hours_per_age requires age > 16; validate inputs first")
            out["capital_net"] = out["capital-gain"] - out["capital-loss"]
            out["hours_per_age"] = out["hours-per-week"] / denominator
        return out


def build_preprocessor(config, engineered):
    low, high = config["preprocessing"]["clip_quantiles"]
    numeric = NUMERIC + (["capital_net", "hours_per_age"] if engineered else [])
    return Pipeline([
        ("clean", AdultCleaner(low, high)),
        ("engineer", FeatureEngineer(engineered)),
        ("encode", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ], remainder="drop", verbose_feature_names_out=True)),
    ])
