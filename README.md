# Green Taxi Fare Prediction

End-to-end machine learning project: predict the fare of a NYC green taxi trip (regression, target `Fare_amount`) and serve the model as a REST API inside Docker.

Data: NYC TLC 2015 Green Taxi Trip Data.

## Results

Models were compared with 5-fold cross-validation on the training set. The best model by CV RMSE was tuned with `RandomizedSearchCV`. The test set was used only for the final report.

| Model | CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Random Forest | 2.51 | 1.25 | 2.51 | 0.914 |
| XGBoost | 2.61 | 1.19 | 2.32 | 0.926 |
| Ridge Regression | 2.76 | 1.48 | 2.76 | 0.895 |

**Final model: tuned Random Forest** (test MAE $1.25, RMSE $2.52, R² 0.913).

Notes:
- The model is chosen by CV score only, never by the test score. XGBoost is better on the test set, but its CV gap to Random Forest (2.51 vs 2.61) is smaller than the fold-to-fold spread (about 0.25 to 0.30), so the selection rule was kept.
- Tuning did not give a meaningful gain (test RMSE 2.52 vs 2.51 before tuning).

## Features

11 inputs: `vendorid`, `rate_code`, pickup and dropoff longitude/latitude, `Passenger_count`, `Trip_distance`, `Trip_type`, `pickup_hour`, `pickup_weekday`.

Data leakage control:
- `Total_amount`, `Tip_amount`, `Tolls_amount`, `Extra`, `MTA_tax`, `Improvement_surcharge`, `Payment_type` and the dropoff time are not used. They are known only after the trip, or are built from the fare itself.
- `Trip_distance` is used on purpose: the user gives an estimated distance when asking for a fare.
- Preprocessing (`SimpleImputer`, `StandardScaler`) is inside the scikit-learn `Pipeline`, so it is fitted on training data only.

## Pipeline (DVC)

All settings live in `params.yaml`. Nothing is hardcoded in the scripts.

| Stage | Script | Output |
|---|---|---|
| make_dataset | `src/data/make_dataset.py` | cleaned data |
| build_features | `src/features/build_features.py` | train/test files |
| train_model | `src/models/train_model.py` | model comparison, best model name |
| tune_model | `src/models/tune_model.py` | tuned model, best params |
| evaluate_model | `src/models/evaluate_model.py` | `reports/metrics.json` |

```bash
pip install -r requirements.txt
dvc repro
```

The raw CSV is tracked with DVC (the `.dvc` pointer file is in Git). No remote storage is configured yet, so place the file at `data/raw/2015_Green_Taxi_Trip_Data.csv` manually.

## API

```bash
uvicorn src.api.main:app --reload
```

- `GET /health` returns `{"status": "ok"}`
- `POST /predict` returns the predicted fare. Wrong values (for example a location outside NYC) are rejected with status 422. The limits come from `params.yaml`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vendorid":2,"rate_code":1,"Pickup_longitude":-73.95,"Pickup_latitude":40.75,"Dropoff_longitude":-73.98,"Dropoff_latitude":40.76,"Passenger_count":1,"Trip_distance":3.5,"Trip_type":1,"pickup_hour":14,"pickup_weekday":2}'
```

Example response: `{"predicted_fare": 17.28, "currency": "USD"}`

## Tests

```bash
pytest -v
```

7 API tests: health check, valid prediction, longer trip costs more, and rejection of wrong vendor, negative distance, location outside NYC and missing field.

## Docker

The model file `models/best_model.joblib` must exist before the build (run `dvc repro` first).

```bash
docker build -t green-taxi-api:local .
docker run --rm -p 8000:8000 green-taxi-api:local
```

## Project structure

```
data/         raw data pointer, interim and processed data (DVC)
models/       trained models (DVC)
reports/      metrics and comparison tables
src/
  data/       cleaning script
  features/   feature building and train/test split
  models/     train, tune, evaluate, predict scripts
  api/        FastAPI app
tests/        API tests
params.yaml   all settings
dvc.yaml      pipeline definition
Dockerfile
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, DVC, FastAPI, pytest, Docker.

## Planned

DVC remote on S3, GitHub Actions CI/CD, deployment on AWS, monitoring.
