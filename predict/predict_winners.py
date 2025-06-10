import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from rich.console import Console
from rich.table import Table

from features.feature_builder import FeatureBuilder
from train.train_model import precision_at_k
from catboost import CatBoostClassifier

MODEL_PATH = Path("models/catboost_gpu.cbm")
FB_PATH = Path("models/feature_builder.joblib")


def load_calendar(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text())
    rows = data.get("data", {}).get("rows", [])
    df = pd.DataFrame(rows)
    df = df[df.get("symbol").notna()]
    df["symbol"] = df["symbol"].str.strip()
    return df


def map_time(value: str) -> str:
    if value == "time-pre-market":
        return "pre"
    if value == "time-after-hours":
        return "post"
    return "intraday"


def build_event_row(ticker: str, release_time: str) -> dict | None:
    tkr = yf.Ticker(ticker)
    try:
        info = tkr.get_info()
        hist = tkr.history(period="1mo")
    except Exception:
        return None
    if hist.empty:
        return None
    prev_prices = hist.iloc[:-1]
    if len(prev_prices) < 20:
        return None
    close_prev = prev_prices["Close"].iloc[-1]
    prev_5d_ret = (prev_prices["Close"].iloc[-1] - prev_prices["Close"].iloc[-5]) / prev_prices["Close"].iloc[-5]
    prev_10d_vol = np.log(prev_prices["Close"]).diff().iloc[-10:].std()
    sma20 = prev_prices["Close"].rolling(20).mean().iloc[-1]
    sma20_gap = (close_prev - sma20) / sma20
    from earnings.build_dataset import rsi
    rsi14 = rsi(prev_prices["Close"]).iloc[-1]
    return {
        "ticker": ticker,
        "release_time": release_time,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "mkt_cap": info.get("marketCap", np.nan) / 1e6 if info.get("marketCap") else np.nan,
        "pe_ratio": info.get("trailingPE", np.nan),
        "beta": info.get("beta", np.nan),
        "prev_qtr_eps": np.nan,
        "eps_est": np.nan,
        "revenue_est": np.nan,
        "prev_5d_ret": prev_5d_ret,
        "prev_10d_vol": prev_10d_vol,
        "sma20_gap": sma20_gap,
        "rsi14": rsi14,
        "short_int_ratio": (info.get("sharesShort") or 0) / info.get("floatShares", np.nan) if info.get("floatShares") else np.nan,
        "label": 0,
    }


def predict_winners(calendar_json: str, top: int) -> None:
    fb = FeatureBuilder.load(FB_PATH)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    cal_df = load_calendar(calendar_json)
    events = []
    for _, row in cal_df.iterrows():
        ticker = row["symbol"].strip()
        release_time = map_time(row.get("time", ""))
        data = build_event_row(ticker, release_time)
        if data:
            events.append(data)
    if not events:
        print("No events to score")
        return
    df = pd.DataFrame(events)
    X, _ = fb.transform(df)
    probs = model.predict_proba(X)[:, 1]
    df["prob_1"] = probs
    df = df.sort_values("prob_1", ascending=False).head(top)

    table = Table(title="Top Earnings Winners")
    for col in ["ticker", "prob_1", "eps_est", "prev_5d_ret"]:
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(r["ticker"], f"{r['prob_1']:.3f}", str(r.get("eps_est", "")), f"{r['prev_5d_ret']:.2%}")
    Console().print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar_json", required=True)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    predict_winners(args.calendar_json, args.top)
