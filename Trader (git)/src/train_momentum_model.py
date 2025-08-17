# scripts/train_model.py
import os, json, argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import joblib

# ---------- CONFIG (edit here or pass via CLI) ----------
DEFAULT_DATASET = "./ml_dataset.csv"
DEFAULT_MODEL   = "./models/momentum_model.pkl"
DEFAULT_META    = "./models/momentum_model.meta.json"
DEFAULT_THRESH  = 0.45  # used by runtime strategy
TIME_SPLIT_COL  = "ts"  # or a datetime col in your dataset; else we’ll fall back to random split
TIME_SPLIT_PCT  = 0.8   # first 80% of time → train; last 20% → test
FEATURES = ["ema_gap", "rsi", "volume_ratio", "bb_width", "above_ema_50"]
TARGET   = "label"
# --------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--model_out", default=DEFAULT_MODEL)
    ap.add_argument("--meta_out", default=DEFAULT_META)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESH)
    ap.add_argument("--time_split", action="store_true", help="Use time-based split by TIME_SPLIT_COL.")
    return ap.parse_args()

def time_based_split(df, frac=0.8, time_col=TIME_SPLIT_COL):
    if time_col not in df.columns:
        return None, None, None, None
    df = df.sort_values(time_col)
    n = len(df)
    cut = int(n * frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]
    return Xtr, Xte, ytr, yte

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    df = pd.read_csv(args.dataset, low_memory=False)
    # ensure numeric
    for col in FEATURES + [TARGET]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 'above_ema_50' to {0,1}
    if "above_ema_50" in df.columns:
        df["above_ema_50"] = pd.to_numeric(df["above_ema_50"], errors="coerce").fillna(0).astype(int)

    # drop incomplete rows
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    # --- Split
    if args.time_split:
        split = time_based_split(df)
        if split[0] is None:
            print(f"[WARN] time column '{TIME_SPLIT_COL}' not found; falling back to random split.")
            X = df[FEATURES]; y = df[TARGET]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        else:
            Xtr, Xte, ytr, yte = split
    else:
        X = df[FEATURES]; y = df[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # --- Model (RF + calibration)
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf = CalibratedClassifierCV(base_rf, method="sigmoid", cv=3)
    clf.fit(Xtr, ytr)

    # --- Evaluation
    yhat = clf.predict(Xte)
    yproba = clf.predict_proba(Xte)[:, 1]
    print("\nConfusion Matrix:")
    print(confusion_matrix(yte, yhat))
    print("\nClassification Report:")
    print(classification_report(yte, yhat, digits=4))
    try:
        print(f"AUC: {roc_auc_score(yte, yproba):.4f}")
        print(f"PR-AUC: {average_precision_score(yte, yproba):.4f}")
    except Exception:
        pass

    # --- Save model + metadata
    joblib.dump(clf, args.model_out)
    meta = {
        "features": FEATURES,
        "threshold": args.threshold,
        "dataset": os.path.abspath(args.dataset),
        "time_split": args.time_split,
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModel saved to {args.model_out}")
    print(f"Meta saved to  {args.meta_out}")

if __name__ == "__main__":
    main()
