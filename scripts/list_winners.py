import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import yfinance as yf

from etl.nasdaq_calendar import fetch_earnings, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List earnings winners")
    parser.add_argument(
        "--date",
        help="Earnings date (YYYY-MM-DD), default: yesterday",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--threshold",
        help="Return threshold for winner",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def get_calendar(date: datetime):
    data = fetch_earnings(date)
    save_json(data, date)
    rows = data.get("data", {}).get("rows", [])
    tickers = [r["symbol"].strip() for r in rows if r.get("symbol")]
    return tickers


def compute_returns(tickers, start, end):
    price_data = yf.download(tickers, start=start, end=end, progress=False)
    if isinstance(price_data, dict) or price_data.empty:
        return {}
    level0 = price_data.columns.get_level_values(0)
    if "Adj Close" in level0:
        closes = price_data["Adj Close"]
    else:
        closes = price_data["Close"]
    returns = {}
    for ticker in closes.columns:
        series = closes[ticker].dropna()
        if series.empty:
            continue
        ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
        returns[ticker] = ret
    return returns

def main():
    args = parse_args()
    target_date = (
        datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.utcnow() - timedelta(days=1)
    )
    tickers = get_calendar(target_date)
    if not tickers:
        print("No tickers found")
        sys.exit(0)
    start = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")
    returns = compute_returns(tickers, start, end)
    winners = {t: r for t, r in returns.items() if r >= args.threshold}
    if not winners:
        print("No winners found")
        sys.exit(0)
    print("Earnings winners:")
    for t, r in sorted(winners.items(), key=lambda x: x[1], reverse=True):
        print(f"{t}: {r:.2%}")


if __name__ == "__main__":
    main()
