# AI4ALL Network Intrusion Demo

Interactive notebook, Streamlit UI, and live packet monitor built around the UNSW-NB15 dataset for the AI4ALL program.

## Contents

- `AI4ALL_Project.ipynb` – exploration, preprocessing, model comparison, feature importance.
- `train_model.py` – trains Logistic Regression, Random Forest, and XGBoost pipelines, evaluates them, and saves artifacts (`artifacts/models.joblib`, `artifacts/metadata.json`). Use `--live-features` to train on the subset of columns obtainable from live capture.
- `app.py` – Streamlit web app to inspect model predictions, see feature importances, and compare models.
- `live_monitor.py` – macOS-friendly scapy-based live monitor that captures packets, extracts flow-level features, and prints anomalies when models flag non-benign flows.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .\\.venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Training Models

> Large `.joblib` artifacts are excluded from Git via `.gitignore`. After cloning (or before your Streamlit session starts) run the training command to regenerate them locally.

```bash
python train_model.py --data-path ./UNSW-NB15/UNSW_NB15_training-set.csv --target label
# For live-monitor alignment / Streamlit deployment:
python train_model.py --data-path ./UNSW-NB15/UNSW_NB15_training-set.csv --target label --live-features
```

Artifacts are written to `artifacts/`. Commit only lightweight files (e.g., `metadata.json`) and regenerate the `.joblib` models as needed.

## Streamlit App

```bash
streamlit run app.py
```

If the artifacts are missing, the app automatically kicks off `python train_model.py --data-path <path> --target label --live-features`. On Streamlit Cloud, set a secret named `DATA_PATH` if your dataset lives in a different location; otherwise place the CSV under `UNSW-NB15/`.

- Left column: load preset benign/attack examples or adjust sliders with explanations.
- Right column: simultaneous predictions from RandomForest, LogisticRegression, and XGBoost.
- Bottom: feature importance chart for the best model.
- Sliders stay within realistic ranges (1st–99th percentile) and show descriptive tooltips.

## Live Monitor

Reads `artifacts/models.joblib` and `metadata.json`, captures packets via scapy, aggregates flow stats, and prints alerts only when a model predicts an attack/high probability.

```bash
sudo python live_monitor.py --ignore-broadcast --ignore-multicast
```

`live_monitor.py` will also auto-train if `artifacts/models.joblib` is absent. Override the dataset path by exporting `DATA_PATH=/path/to/UNSW_NB15_training-set.csv` before running.

Key flags:

- `--threshold 0.99` (default) – probability needed to alert.
- `--min-pkts`, `--min-bytes` – skip tiny flows.
- `--cooldown 60` – seconds before re-alerting on the same flow/model.
- `--iface auto` – auto-detects macOS en0/eth0. Override as needed.

Alerts show model, flow tuple, predicted label, probability, packet rate, counts, and duration.

> **Tip:** Retrain with `--live-features` for far fewer false positives when running the live monitor.

## Notes

- The live extractor only has access to a subset of UNSW features; models trained on full data will raise many benign alerts. Use the live-feature training mode to align train/test.
- Packet capture on macOS requires root privileges (`sudo`). If you have Wireshark/tshark installed you could adapt the monitor to pyshark, but scapy keeps dependencies minimal.
