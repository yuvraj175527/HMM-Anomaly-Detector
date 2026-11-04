# Machine Anomaly Detection Using Hidden Markov Models (HMMs)

A Python project that learns normal machine-sensor behaviour with a Gaussian Hidden Markov Model (HMM), then flags unusual time windows using sequence log-likelihood.

## Features
- Simulated normal and test machine sensor streams (including injected anomalies)
- Gaussian HMM training with `hmmlearn`
- Hidden-state selection using BIC across candidate state counts
- Training-window log-likelihood threshold
- Sliding-window anomaly detection
- Plot with detected anomaly windows highlighted
- Optional Streamlit dashboard

## Project structure
```text
HMM-Anomaly-Detector/
├── dataset/                 # Generated CSV datasets
├── models/                  # Trained model and metadata
├── src/
│   ├── data.py
│   ├── train.py
│   ├── detect.py
│   └── visualize.py
├── app.py                   # Optional Streamlit dashboard
├── generate_data.py
├── requirements.txt
└── README.md
```

## 1. Setup
Python 3.10 or 3.11 is recommended.

### Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Generate the sample sensor data
```bash
python generate_data.py
```
This creates:
- `dataset/normal_sensor_data.csv` — normal operating data for training
- `dataset/test_sensor_data.csv` — a test stream with injected abnormal periods

The generated data is synthetic and intended for demonstrating the assignment workflow. Replace it with instructor-provided sensor data when available.

## 3. Train the HMM
```bash
python -m src.train
```
The training script tests several hidden-state counts and selects the lowest-BIC candidate. It saves the selected Gaussian HMM and training threshold metadata under `models/`.

## 4. Detect anomalies
```bash
python -m src.detect
```
The detector scores overlapping sliding windows using average log-likelihood per observation. Windows below the training-derived threshold are flagged. Results are saved to `outputs/detections.csv`.

## 5. Visualize results
```bash
python -m src.visualize
```
The plot is saved to `outputs/anomaly_detection.png`.

## 6. Optional interactive dashboard
```bash
streamlit run app.py
```
Open the local URL printed by Streamlit.

## Method and design choices
- **Emission model:** Gaussian emissions (`GaussianHMM`) for continuous-valued sensor readings.
- **Hidden states:** BIC is used to compare candidate state counts (2–5) on the normal training sequence. The selected count is recorded in `models/model_metadata.json`.
- **Threshold:** The lower 1st percentile of normal training-window average log-likelihoods is used as the anomaly threshold. This is a configurable false-alarm trade-off, not a universal constant.
- **Sliding window:** A fixed window of 50 samples with a stride of 10 samples is used by default. Adjust these in `src/detect.py` to match the sampling rate and expected anomaly duration.
- **Evaluation:** The test CSV includes a `label` column (0 = normal, 1 = injected anomaly) so precision, recall, F1 and ROC-AUC can be calculated for this demonstration.

## Important notes
- HMM likelihoods are sequence-level scores. This project uses average log-likelihood per observation to make windows of equal length comparable.
- The provided data is simulated; results on real industrial data may differ.
- A Gaussian HMM assumes observations are generated from a mixture of state-dependent Gaussian distributions and that hidden-state transitions follow a Markov process.

## Author
Add your name and course details here.
