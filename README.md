# Customer Churn

A small FastAPI service and training pipeline for predicting telco customer churn using scikit-learn.

## Contents
- Dataset: [data/churn.csv](data/churn.csv)  
- Training script: [src/train.py](src/train.py) (see [`FILE_NAME`](src/train.py), [`preprocessor`](src/train.py]) )
- Serving app: [src/serve.py](src/serve.py) (serves [`TelcoCustomer`](src/serve.py), loads [`pipeline`](src/serve.py) and [`label_encoder`](src/serve.py))
- Project metadata: [pyproject.toml](pyproject.toml)

## Quickstart

1. Create & activate a Python >= 3.12 virtual environment.
2. Install dependencies:
   - `pip install -r <dependencies from pyproject.toml>` or use the list from [pyproject.toml](pyproject.toml).
3. Train a model:
   - Run the training script: `python src/train.py`
   - This script reads [data/churn.csv](data/churn.csv) (see [`FILE_NAME`](src/train.py)) and writes model artifacts (e.g., `pipeline.pkl`, `label_encoder.pkl`).

4. Serve the API:
   - Ensure `pipeline.pkl` and `label_encoder.pkl` are present at the working directory (the app expects these names).
   - Start uvicorn: `uvicorn src.serve:app --reload`
   - Open `http://127.0.0.1:8000/docs` for the interactive API docs.

## API

The FastAPI app is implemented in [src/serve.py](src/serve.py). Key items:
- Request model: [`TelcoCustomer`](src/serve.py) — Pydantic model describing input features.
- Endpoints:
  - `GET /` — health/info endpoint (see [`root`](src/serve.py)).
  - `POST /predict` — accepts [`TelcoCustomer`](src/serve.py) and returns churn prediction (see [`predict`](src/serve.py)). The app loads [`pipeline`](src/serve.py) and [`label_encoder`](src/serve.py) on startup.

## Reproduce training

The training script [src/train.py](src/train.py) performs:
- CSV reading and numeric coercion for `['TotalCharges','MonthlyCharges','tenure']` (see [`FILE_NAME`](src/train.py)).
- Simple preprocessing via a `ColumnTransformer` (`preprocessor`) and a `RandomForestClassifier`.
- Persists artifacts with `joblib` (e.g., `pipeline.pkl`, `label_encoder.pkl`).

Run:
- `python src/train.py`
