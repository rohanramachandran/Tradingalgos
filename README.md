This project aims to build a reproducible pipeline that collects U.S. earnings releases, engineers predictive features and produces a daily ranked list of "earnings winners". The repository is organised as follows:

```
data/      - local storage for raw and processed data
etl/       - data collection scripts
features/  - feature engineering code
models/    - model training and inference
backtest/  - strategy evaluation utilities
```

### Quick start

1. Copy `.env.example` to `.env` and supply API keys for Nasdaq and EODHD.
2. Install dependencies (requires Python 3.11):

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Nasdaq earnings calendar ETL for the current day:

   ```bash
   python etl/nasdaq_calendar.py
   ```

Raw JSON files will be saved in `data/raw/`.
