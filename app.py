from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.data import ROOT, FEATURES

st.set_page_config(page_title="HMM Anomaly Detector", layout="wide")
st.title("Machine Anomaly Detector")
st.caption("Gaussian Hidden Markov Model with sliding-window likelihood scoring")

model_path = ROOT / "models" / "hmm_model.pkl"
meta_path = ROOT / "models" / "model_metadata.json"
test_path = ROOT / "dataset" / "test_sensor_data.csv"

if not model_path.exists() or not meta_path.exists():
    st.warning("Model not found. Run `python generate_data.py` and `python -m src.train` first.")
    st.stop()

bundle = joblib.load(model_path)
metadata = json.loads(meta_path.read_text())
model, scaler = bundle["model"], bundle["scaler"]
window_size = int(metadata["window_size"])
threshold = float(metadata["anomaly_threshold"])

uploaded = st.file_uploader("Upload sensor CSV (temperature, rpm, vibration)", type="csv")
if uploaded:
    frame = pd.read_csv(uploaded)
elif test_path.exists():
    frame = pd.read_csv(test_path)
    st.info("Showing the bundled simulated test stream.")
else:
    st.stop()

if not set(FEATURES).issubset(frame.columns):
    st.error(f"CSV must contain: {', '.join(FEATURES)}")
    st.stop()

x = scaler.transform(frame[FEATURES])
stride = 10
flags = np.zeros(len(frame), dtype=int)
window_scores = []
for start in range(0, len(x) - window_size + 1, stride):
    end = start + window_size
    score = model.score(x[start:end]) / window_size
    is_anomaly = score < threshold
    window_scores.append(score)
    if is_anomaly:
        flags[start:end] = 1

frame["predicted_anomaly"] = flags
c1, c2, c3 = st.columns(3)
c1.metric("Samples", len(frame))
c2.metric("Flagged samples", int(flags.sum()))
c3.metric("Threshold", f"{threshold:.3f}")

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(frame.index, frame["temperature"], label="Temperature")
mask = frame["predicted_anomaly"].astype(bool)
ax.scatter(frame.index[mask], frame.loc[mask, "temperature"], s=12, label="Detected anomaly")
ax.set_xlabel("Time step")
ax.set_ylabel("Temperature")
ax.set_title("Sensor readings and detected anomaly regions")
ax.legend()
ax.grid(alpha=0.25)
st.pyplot(fig)
st.dataframe(frame[frame["predicted_anomaly"] == 1].head(100), use_container_width=True)
