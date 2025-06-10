This project aims to build a reproducible pipeline that collects U.S. earnings releases, engineers predictive features and produces a daily ranked list of "earnings winners". The repository is organised as follows:

```
data/      - local storage for raw and processed data
etl/       - data collection scripts
features/  - feature engineering code
models/    - model training and inference
backtest/  - strategy evaluation utilities
```

### Quick start

1. (Optional) copy `.env.example` to `.env` to customise paths.
2. Install dependencies (requires Python 3.11):

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Nasdaq earnings calendar ETL for the current day:

   ```bash
   python etl/nasdaq_calendar.py
   ```

Raw JSON files will be saved in `data/raw/`.

### Full pipeline example

After fetching the Nasdaq calendar JSON, run the entire process:

```bash
python run_pipeline.py \
    --start 2018-01-01 --end 2025-05-31 \
    --calendar_json calendar_20250610.json \
    --top 5 --retrain
```

This builds/updates the dataset, trains the CatBoost model on GPU and prints the top candidates for the next session.
