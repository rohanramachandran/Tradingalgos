import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm


OUT_PATH = Path("data/earnings_dataset.parquet")
CHUNK_SIZE = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build earnings events dataset")
    p.add_argument("--start", required=True, type=str, help="start date YYYY-MM-DD")
    p.add_argument("--end", required=True, type=str, help="end date YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="rebuild even if file exists")
    return p.parse_args()


import functools, pathlib, pickle, time, pandas as pd, requests

_UNIVERSE_CACHE = pathlib.Path("data/universe_cache.pkl")

@functools.lru_cache(maxsize=1)
def _scrape_wiki(url: str, col: str) -> list[str]:
    """Return the ticker column from the first HTML table on the page."""
    html = requests.get(url, timeout=15).text
    return pd.read_html(html)[0][col].str.strip().tolist()

def get_universe() -> list[str]:
    """
    Union of S&P 500 / 400 / 600 tickers, scraped from Wikipedia and
    cached on disk for 24 h to avoid repeated HTTP traffic.
    """
    if _UNIVERSE_CACHE.exists() and time.time() - _UNIVERSE_CACHE.stat().st_mtime < 86_400:
        return pickle.loads(_UNIVERSE_CACHE.read_bytes())

    sp500 = _scrape_wiki(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "Symbol",
    )
    sp400 = _scrape_wiki(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "Symbol",
    )
    sp600 = _scrape_wiki(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "Symbol",
    )

    universe = sorted({*sp500, *sp400, *sp600})
    _UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _UNIVERSE_CACHE.write_bytes(pickle.dumps(universe))
    return universe


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window).mean()
    loss = down.rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def infer_release_time(ts: pd.Timestamp) -> str:
    if ts is pd.NaT or ts.hour == 0 and ts.minute == 0:
        return "intraday"
    if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
        return "pre"
    if ts.hour >= 16:
        return "post"
    return "intraday"


def process_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> list:
    tkr = yf.Ticker(ticker)
    try:
        cal = tkr.get_earnings_dates(limit=60)
    except Exception:
        return []
    if cal is None or cal.empty:
        return []
    cal = cal[(cal.index >= start) & (cal.index <= end)]
    if cal.empty:
        return []
    try:
        info = tkr.get_info()
    except Exception:
        info = {}
    try:
        hist = tkr.history(start=start - timedelta(days=40), end=end + timedelta(days=5))
    except Exception:
        return []
    if hist.empty:
        return []
    dollar_vol = (hist["Close"] * hist["Volume"]).median()
    if dollar_vol < 10_000_000:
        return []
    earnings = tkr.get_earnings()
    rows = []
    for dt, row in cal.iterrows():
        prev_prices = hist[:dt - timedelta(days=1)]
        if len(prev_prices) < 21:
            continue
        future_prices = hist[dt: dt + timedelta(days=2)]
        if len(future_prices) < 3:
            continue
        close_pre = prev_prices["Close"].iloc[-1]
        close_plus2 = future_prices["Close"].iloc[2]
        label = int((close_plus2 - close_pre) / close_pre >= 0.05)
        prev_5d_ret = (prev_prices["Close"].iloc[-1] - prev_prices["Close"].iloc[-5]) / prev_prices["Close"].iloc[-5]
        prev_10d_vol = np.log(prev_prices["Close"]).diff().iloc[-10:].std()
        sma20 = prev_prices["Close"].rolling(20).mean().iloc[-1]
        sma20_gap = (prev_prices["Close"].iloc[-1] - sma20) / sma20
        rsi14 = rsi(prev_prices["Close"]).iloc[-1]
        eps_est = row.get("epsestimate") or row.get("EPS Estimate") or np.nan
        revenue_est = row.get("revenueestimate") or row.get("Revenue Estimate") or np.nan
        release_time = infer_release_time(dt)
        prev_eps = np.nan
        if earnings is not None and not earnings.empty:
            prev_eps = earnings.iloc[-1]["Earnings"]
        rows.append({
            "ticker": ticker,
            "event_date": dt.to_pydatetime(),
            "release_time": release_time,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "mkt_cap": info.get("marketCap", np.nan) / 1e6 if info.get("marketCap") else np.nan,
            "pe_ratio": info.get("trailingPE", np.nan),
            "beta": info.get("beta", np.nan),
            "prev_qtr_eps": prev_eps,
            "eps_est": eps_est,
            "revenue_est": revenue_est,
            "prev_5d_ret": prev_5d_ret,
            "prev_10d_vol": prev_10d_vol,
            "sma20_gap": sma20_gap,
            "rsi14": rsi14,
            "short_int_ratio": (info.get("sharesShort") or 0) / info.get("floatShares", np.nan) if info.get("floatShares") else np.nan,
            "label": label,
        })
    return rows


def build_dataset(start: str, end: str) -> Path:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    tmp_dir = Path("data/parquet")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    universe = get_universe()
    all_rows = []
    chunk_idx = 0
    for ticker in tqdm(universe, desc="Tickers"):
        rows = process_ticker(ticker, start_dt, end_dt)
        if not rows:
            continue
        all_rows.extend(rows)
        if len(all_rows) >= CHUNK_SIZE:
            df = pd.DataFrame(all_rows)
            chunk_file = tmp_dir / f"chunk_{chunk_idx}.parquet"
            df.to_parquet(chunk_file, index=False)
            all_rows = []
            chunk_idx += 1
    if all_rows:
        df = pd.DataFrame(all_rows)
        chunk_file = tmp_dir / f"chunk_{chunk_idx}.parquet"
        df.to_parquet(chunk_file, index=False)
    files = list(tmp_dir.glob("chunk_*.parquet"))
    df_list = [pd.read_parquet(f) for f in files]
    full = pd.concat(df_list, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PATH, index=False)
    for f in files:
        f.unlink()
    return OUT_PATH


if __name__ == "__main__":
    args = parse_args()
    if OUT_PATH.exists() and not args.force:
        print(f"Dataset already exists at {OUT_PATH}")
    else:
        path = build_dataset(args.start, args.end)
        print(f"Saved dataset to {path}")
