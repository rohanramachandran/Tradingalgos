import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests


BASE_URL = "https://api.nasdaq.com/api/calendar/earnings"
EODHD_URL = "https://eodhd.com/api/calendar/earnings"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}


def fetch_earnings(date: datetime) -> dict:
    """Fetch earnings calendar for a specific date from Nasdaq."""
    params = {"date": date.strftime("%Y-%m-%d")}
    response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
    if response.status_code == 200:
        return response.json()
    # fallback to EODHD
    token = os.getenv("EODHD_API_KEY")
    if not token:
        response.raise_for_status()
    alt_params = {
        "api_token": token,
        "from": date.strftime("%Y-%m-%d"),
        "to": date.strftime("%Y-%m-%d"),
    }
    alt_resp = requests.get(EODHD_URL, params=alt_params, timeout=30)
    alt_resp.raise_for_status()
    return alt_resp.json()


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
