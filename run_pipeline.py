import argparse
from pathlib import Path

from earnings.build_dataset import build_dataset, OUT_PATH as DATA_PATH
from train.train_model import train_model
from predict.predict_winners import predict_winners


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run earnings winner pipeline")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--calendar_json", required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        build_dataset(args.start, args.end)
    else:
        if args.retrain:
            build_dataset(args.start, args.end)

    model_path = Path("models/catboost_gpu.cbm")
    if not model_path.exists() or args.retrain:
        train_model()

    predict_winners(args.calendar_json, args.top)
