# CyberShield IDS

A 3-layer machine-learning intrusion detection & autonomous response system. It classifies
network traffic flows and recommends a containment action: `ALLOW → MONITOR → BLOCK → ISOLATE`.

## Pipeline architecture

| Layer | Component | Purpose |
|---|---|---|
| 1 | Stacked Random Forest + XGBoost ensemble (meta-classifier) | Supervised attack/benign classification |
| 2 | Autoencoder | Unsupervised anomaly detection via reconstruction error (MSE) |
| 3 | PPO (stable-baselines3) reinforcement-learning agent | Picks the final containment action from `[prediction, mse, severity]` |

Trained and evaluated on the **CICIDS2017** dataset (Friday DDoS capture used for deployment testing).

## Dataset

This repo does **not** include the CICIDS2017 CSVs (several files exceed GitHub's size limits,
and the dataset is a third-party benchmark, not something to redistribute here).

Download it directly from the official source — Canadian Institute for Cybersecurity, University
of New Brunswick:

**https://www.unb.ca/cic/datasets/ids-2017.html**

After downloading, place the CSVs in a `MachineLearningCVE/` folder at the project root (this
path is already excluded via `.gitignore`).

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`.

## Repository contents

- `app.py` — Streamlit dashboard
- `cyber breach (no LSTM_DNN) UPDATE 17th April .ipynb` — training notebook (source of truth for model training)
- `deployment_notebook(no_training_,_final_ui_ux).py` — deployment evaluation script
- `rf_final.pkl`, `xgb_final.pkl`, `meta_final.pkl`, `scaler_final.pkl`, `threshold_final.npy`, `autoencoder_final.h5`, `rl_model_final/` — trained model artifacts produced by the notebook
- `requirements.txt`, `runtime.txt` — dependencies / runtime version
