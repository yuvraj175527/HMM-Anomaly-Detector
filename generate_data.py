from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
DATA.mkdir(exist_ok=True)

def make_normal(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    # Two normal operating regimes with smooth state changes.
    states = np.zeros(n, dtype=int)
    for i in range(1, n):
        if states[i - 1] == 0:
            states[i] = 1 if rng.random() < 0.035 else 0
        else:
            states[i] = 0 if rng.random() < 0.08 else 1

    means = np.array([[65.0, 1200.0, 0.20], [72.0, 1450.0, 0.28]])
    stds = np.array([[1.8, 45.0, 0.025], [2.2, 55.0, 0.035]])
    x = np.vstack([rng.normal(means[s], stds[s]) for s in states])
    return pd.DataFrame(x, columns=["temperature", "rpm", "vibration"])

def main():
    normal = make_normal()
    normal.to_csv(DATA / "normal_sensor_data.csv", index=False)

    test = make_normal(n=1800, seed=123)
    test["label"] = 0
    # Inject three abnormal intervals with higher temperature/vibration and unusual RPM.
    for start, end in [(350, 410), (900, 970), (1450, 1525)]:
        test.loc[start:end, "temperature"] += 12
        test.loc[start:end, "rpm"] -= 260
        test.loc[start:end, "vibration"] += 0.22
        test.loc[start:end, "label"] = 1
    test.to_csv(DATA / "test_sensor_data.csv", index=False)
    print("Created normal_sensor_data.csv and test_sensor_data.csv in dataset/")

if __name__ == "__main__":
    main()
