from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.data import ROOT, FEATURES, load_test_data

MODEL_PATH = ROOT / "models" / "hmm_model.pkl"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def sliding_window_scores(model, x, window_size=50, stride=10):
    rows = []
    for start in range(0, len(x) - window_size + 1, stride):
        end = start + window_size
        score = model.score(x[start:end]) / window_size
        rows.append({"start": start, "end": end, "score": score})
    return pd.DataFrame(rows)

def main():
    bundle = joblib.load(MODEL_PATH)
    model, scaler = bundle["model"], bundle["scaler"]
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    test = load_test_data()
    x = scaler.transform(test[FEATURES])

    scores = sliding_window_scores(model, x, metadata["window_size"], stride=10)
    threshold = metadata["anomaly_threshold"]
    scores["anomaly"] = (scores["score"] < threshold).astype(int)

    # Map window decisions to samples; a sample is anomalous if covered by any flagged window.
    sample_flags = np.zeros(len(test), dtype=int)
    for row in scores.itertuples():
        if row.anomaly:
            sample_flags[row.start:row.end] = 1

    test["anomaly_score"] = np.nan
    test["predicted_anomaly"] = sample_flags
    test.to_csv(OUTPUT_DIR / "detections.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "window_scores.csv", index=False)

    y_true = test["label"].to_numpy()
    print("Precision:", round(precision_score(y_true, sample_flags, zero_division=0), 4))
    print("Recall:", round(recall_score(y_true, sample_flags, zero_division=0), 4))
    print("F1:", round(f1_score(y_true, sample_flags, zero_division=0), 4))
    if len(np.unique(y_true)) == 2:
        print("ROC-AUC (using binary flags):", round(roc_auc_score(y_true, sample_flags), 4))
    print("Saved outputs/detections.csv and outputs/window_scores.csv")

if __name__ == "__main__":
    main()
