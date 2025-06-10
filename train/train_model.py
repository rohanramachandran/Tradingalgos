import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from features.feature_builder import FeatureBuilder

DATA_PATH = Path("data/earnings_dataset.parquet")
MODEL_PATH = Path("models/catboost_gpu.cbm")
FB_PATH = Path("models/feature_builder.joblib")

PARAMS = dict(
    depth=8,
    learning_rate=0.05,
    iterations=2000,
    l2_leaf_reg=3,
    random_strength=1,
    loss_function="Logloss",
    task_type="GPU",
    devices="0",
    early_stopping_rounds=200,
    verbose=100,
    random_seed=42,
)


def precision_at_k(y_true, y_prob, pct=0.05):
    k = max(1, int(len(y_prob) * pct))
    top_idx = np.argsort(y_prob)[-k:][::-1]
    return (y_true.iloc[top_idx] == 1).mean()


def train_model() -> None:
    df = pd.read_parquet(DATA_PATH).sort_values("event_date")
    fb = FeatureBuilder()
    fb.fit(df)
    X, y = fb.transform(df)
    split_date = df["event_date"].max() - pd.DateOffset(months=18)
    train_idx = df["event_date"] < split_date
    test_idx = df["event_date"] >= split_date
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    tscv = TimeSeriesSplit(n_splits=5)
    best_iter = PARAMS["iterations"]
    for tr, val in tscv.split(X_train):
        tr_pool = Pool(X_train.iloc[tr], y_train.iloc[tr])
        val_pool = Pool(X_train.iloc[val], y_train.iloc[val])
        model = CatBoostClassifier(**PARAMS)
        model.fit(tr_pool, eval_set=val_pool)
        best_iter = model.get_best_iteration()
    final_model = CatBoostClassifier(**{**PARAMS, "iterations": best_iter})
    final_model.fit(Pool(X_train, y_train))
    fb.save(FB_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_PATH)

    prob_tr = final_model.predict_proba(X_train)[:, 1]
    prob_te = final_model.predict_proba(X_test)[:, 1]
    metrics = pd.DataFrame({
        "AUROC": [roc_auc_score(y_train, prob_tr), roc_auc_score(y_test, prob_te)],
        "Acc": [accuracy_score(y_train, prob_tr > 0.5), accuracy_score(y_test, prob_te > 0.5)],
        "P@Top1": [precision_at_k(y_train, prob_tr, 0.01), precision_at_k(y_test, prob_te, 0.01)],
        "P@Top5": [precision_at_k(y_train, prob_tr, 0.05), precision_at_k(y_test, prob_te, 0.05)],
    }, index=["Train", "Test"])
    print(metrics)


if __name__ == "__main__":
    train_model()
