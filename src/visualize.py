from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.data import ROOT

def main():
    data = pd.read_csv(ROOT / "outputs" / "detections.csv")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data.index, data["temperature"], label="Temperature", linewidth=1)
    anomaly = data["predicted_anomaly"].astype(bool)
    ax.scatter(data.index[anomaly], data.loc[anomaly, "temperature"],
               label="Detected anomaly", s=12)
    # Shade contiguous detected regions.
    in_region = False
    start = 0
    for i, flag in enumerate(anomaly.tolist() + [False]):
        if flag and not in_region:
            start, in_region = i, True
        elif not flag and in_region:
            ax.axvspan(start, i - 1, alpha=0.18)
            in_region = False
    ax.set_title("Machine Sensor Anomaly Detection (Gaussian HMM)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Temperature")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ROOT / "outputs" / "anomaly_detection.png"
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")
    plt.show()

if __name__ == "__main__":
    main()
