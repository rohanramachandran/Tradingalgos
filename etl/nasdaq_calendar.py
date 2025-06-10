import json
from datetime import datetime, timezone
from pathlib import Path
import requests


BASE_URL = "https://api.nasdaq.com/api/calendar/earnings"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}


def fetch_earnings(date: datetime) -> dict:
    """Fetch earnings calendar for a specific date from Nasdaq."""
    params = {"date": date.strftime("%Y-%m-%d")}
    response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def save_json(data: dict, date: datetime, out_dir: str = "data/raw") -> Path:
    dt = date.strftime("%Y-%m-%d")
    out_path = Path(out_dir) / f"nasdaq_{dt}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return out_path


if __name__ == "__main__":
    target_date = datetime.now(timezone.utc).astimezone().date()
    data = fetch_earnings(target_date)
    path = save_json(data, target_date)
    print(f"Saved {path}")
