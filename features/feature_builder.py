from __future__ import annotations
import joblib
from pathlib import Path
import pandas as pd
import numpy as np


NUM_COLS = [
    "mkt_cap",
    "pe_ratio",
    "beta",
    "prev_qtr_eps",
    "eps_est",
    "revenue_est",
    "prev_5d_ret",
    "prev_10d_vol",
    "sma20_gap",
    "rsi14",
    "short_int_ratio",
]
CAT_COLS = ["sector", "industry", "release_time"]


class FeatureBuilder:
    def __init__(self):
        self.quantiles = {}
        self.medians = {}
        self.means = {}
        self.stds = {}
        self.target_enc = {}

    def fit(self, df: pd.DataFrame) -> None:
        for c in NUM_COLS:
            low = df[c].quantile(0.005)
            high = df[c].quantile(0.995)
            self.quantiles[c] = (low, high)
            self.medians[c] = df[c].median()
            self.means[c] = df[c].mean()
            self.stds[c] = df[c].std() or 1.0
        for c in CAT_COLS:
            self.target_enc[c] = df.groupby(c)["label"].mean()

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        out = df.copy()
        for c in NUM_COLS:
            low, high = self.quantiles[c]
            out[c] = out[c].clip(low, high)
            out[c] = out[c].fillna(self.medians[c])
            out[c] = (out[c] - self.means[c]) / self.stds[c]
        for c in CAT_COLS:
            mapping = self.target_enc.get(c, {})
            out[c] = out[c].map(mapping).fillna(mapping.mean() if len(mapping) else 0)
        y = out.pop("label")
        return out, y

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FeatureBuilder":
        return joblib.load(path)
