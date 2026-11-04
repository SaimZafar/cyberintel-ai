# CyberIntel AI: Detection Engine

CyberIntel AI is a mobile app that sends security analysts network threat alerts and explains each one. This repository holds the AI engine behind it.

The engine takes one network flow and returns a complete decision through a single endpoint, `POST /detect`:

- the attack category (Normal or one of nine UNSW-NB15 attack types) and the model's confidence
- an anomaly score, from an Isolation Forest trained only on normal traffic
- a severity score from 0 to 100 and a level (Low, Medium, High, Critical)
- the top reasons behind the decision (SHAP), written as plain sentences and checked automatically against the event's real values
- suggested investigation steps

The classifier is an XGBoost model with square-root class weights, trained on UNSW-NB15 with a group-aware split so that identical flows never sit in both training and validation. The service is built with FastAPI and is called by the app's backend. The Flutter app and backend live in a separate repository.

## Data (not included)

The dataset is not in this repository. Download the two partition files of UNSW-NB15 from the official UNSW page (or a copy of the same files) and place them here:

```
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
```

Use the full version with 45 columns (it contains `id`, `sttl` and `ct_state_ttl`). Some copies online have only 34 features.

## Install

Python 3.12 or newer, then from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The service itself needs only: fastapi, uvicorn, pydantic, pandas, numpy, scikit-learn, joblib, xgboost and shap. PyTorch and TabNet are used only by experiment scripts.

## Run the service

The trained models are included in `models/`, so you can start the service without any data:

```powershell
python -m uvicorn service.main:app --port 8000
```

Wait for "Application startup complete" (the first start takes 10 to 30 seconds). Then:

- `GET /health` returns the status and model version.
- `POST /detect` takes one flow as JSON and returns the decision.
- Interactive docs: http://127.0.0.1:8000/docs

**Input.** A JSON object with the raw UNSW-NB15 flow fields, for example `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload` and the `ct_*` counters. Missing numeric fields default to 0 and are listed in `missing_fields`. A non-numeric value in a numeric field returns HTTP 422.

**Output (main fields).**

| Field | Meaning |
|---|---|
| `attack_category` | Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode or Worms |
| `confidence` | Classifier confidence, 0 to 1 |
| `anomaly_score` | 0 to 1, how unusual the flow is compared with normal traffic |
| `severity_score`, `severity_level` | 0 to 100, and Low / Medium / High / Critical |
| `severity_breakdown` | The three terms of the severity formula |
| `top_features` | The main reasons, each with a plain-language sentence |
| `explanation_text`, `suggested_steps` | Text for the analyst |
| `model_version`, `processing_ms` | Version and time spent in the engine |

See `docs/sample_responses.json` for real example responses with every field.

Typical engine time per call is about <!-- CHECK: confirm against reports/logs/profile_engine.txt --> ms.

## Rebuild the models

With the data in place, run the pipeline from the project folder (close other heavy apps first, it takes roughly 20 to 30 minutes on 8 GB of RAM):

```powershell
python src\run_all.py
```

It runs, in order: splits, preprocessing, baseline, model comparison, class-weight sweep, duplicate test, XGBoost tuning, confusion matrix, anomaly detector, severity evaluation and demo-pool evaluation. Each step saves its output to `reports/logs/`. If a step fails, restart from it, for example `python src\run_all.py anomaly`. <!-- CHECK: alert_config.json is not made by run_all.py; the committed copy in models/ is the one the service uses -->

## Folder map

| Path | Contents |
|---|---|
| `service/main.py` | The FastAPI service |
| `src/engine.py`, `features.py`, `explain.py`, `explanation.py` | Runtime code used by the service |
| `models/` | Trained classifier, anomaly forest and reference, and the JSON configs |
| `src/make_splits.py`, `prepare_data.py`, `train_models.py`, `tune_xgb.py`, `anomaly.py` | Data preparation and training |
| `docs/` | Example `/detect` responses |
| `reports/` | Result tables, figures and logs |

## Experiments

`src/` also contains comparison and evaluation scripts (other model families, validation protocols, shift and threshold studies). Their outputs are in `reports/`. The service does not need any of them.

## Limitations

- UNSW-NB15 is a lab dataset with many duplicate flows and some identical flows carrying different labels, so no model can reach perfect accuracy on it.
- Analysis, Backdoor and DoS are often confused with Exploits. The explanations show what the model relied on, not proof of malicious intent.
- Performance drops on the official test partition compared with validation, so numbers from validation alone should not be quoted as expected real-world accuracy.