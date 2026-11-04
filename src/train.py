from pathlib import Path
import json
import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from src.data import ROOT, FEATURES, load_training_data, fit_scaler

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def bic_score(model, x):
    # Approximate parameter count for a full-covariance Gaussian HMM.
    n_states, n_features = model.n_components, x.shape[1]
    n_params = (n_states - 1) + n_states * (n_states - 1)
    n_params += n_states * n_features
    n_params += n_states * n_features * (n_features + 1) / 2
    log_likelihood = model.score(x)
    return -2 * log_likelihood + n_params * np.log(len(x))

def main():
    frame = load_training_data()
    x, scaler = fit_scaler(frame)

    candidates = []
    for states in range(2, 6):
        model = GaussianHMM(
            n_components=states,
            covariance_type="full",
            n_iter=300,
            tol=1e-3,
            random_state=42,
            min_covar=1e-3,
        )
        model.fit(x)
        candidates.append((bic_score(model, x), states, model))

    bic, selected_states, best_model = min(candidates, key=lambda item: item[0])

    # Score non-overlapping training chunks to form a normal likelihood distribution.
    window = 50
    train_scores = []
    for start in range(0, len(x) - window + 1, window):
        chunk = x[start:start + window]
        train_scores.append(best_model.score(chunk) / len(chunk))

    threshold = float(np.percentile(train_scores, 1))
    joblib.dump({"model": best_model, "scaler": scaler, "features": FEATURES}, MODEL_DIR / "hmm_model.pkl")
    metadata = {
        "selected_hidden_states": int(selected_states),
        "bic": float(bic),
        "window_size": window,
        "threshold_percentile": 1,
        "anomaly_threshold": threshold,
        "training_window_count": len(train_scores),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Training complete.")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
