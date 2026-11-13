# Green Taxi Fare Prediction

End-to-end machine learning project that predicts the fare of a NYC green taxi trip (regression, target `Fare_amount`) and serves predictions through a FastAPI REST API packaged with Docker.

**Dataset:** NYC TLC 2015 Green Taxi Trip Data.

## Model results

Models were compared using 5-fold cross-validation on the training set. The model-selection rule uses cross-validation RMSE, not test-set performance. The selected Random Forest was then tuned with `RandomizedSearchCV`; the test set was reserved for final evaluation.

### Model comparison

| Model | CV RMSE Mean | CV RMSE Std |
|---|---:|---:|
| Random Forest | 2.5064 | 0.2470 |
| XGBoost | 2.6075 | 0.2977 |
| Ridge Regression | 2.7559 | 0.2901 |

### Final tuned model

- **Selected model:** Tuned Random Forest
- **Best tuning CV RMSE:** 2.5417
- **Test MAE:** 1.2529
- **Test RMSE:** 2.5205
- **Test R²:** 0.9130

The model is selected using cross-validation only. XGBoost had better test metrics in the comparison, but the selection rule was intentionally not changed based on test-set results.

## Features and leakage control

The model uses 11 input features: `vendorid`, `rate_code`, pickup and dropoff longitude/latitude, `Passenger_count`, `Trip_distance`, `Trip_type`, `pickup_hour`, and `pickup_weekday`.

The following fields are excluded to prevent target leakage because they are only known after the trip or are derived from the fare: `Total_amount`, `Tip_amount`, `Tolls_amount`, `Extra`, `MTA_tax`, `Improvement_surcharge`, `Payment_type`, and dropoff time.

`Trip_distance` is included because an estimated distance can be supplied when requesting a fare. Preprocessing (imputation and scaling) is contained in the scikit-learn pipeline and is fitted on training data only.

## ML pipeline (DVC)

All pipeline settings are stored in `params.yaml`.

| Stage | Script | Main output |
|---|---|---|
| `make_dataset` | `src/data/make_dataset.py` | Cleaned dataset |
| `build_features` | `src/features/build_features.py` | Train/test datasets |
| `train_model` | `src/models/train_model.py` | Model comparison and selected model |
| `tune_model` | `src/models/tune_model.py` | Tuned model and best parameters |
| `evaluate_model` | `src/models/evaluate_model.py` | `reports/metrics.json` |

The raw dataset and model artifacts are tracked with DVC. The repository configures an S3 DVC remote; access to that remote requires suitable AWS credentials.

To reproduce the pipeline in an environment with the required dependencies and AWS access:

```bash
pip install -r requirements.txt
dvc pull
dvc repro
```

## API

The FastAPI application exposes:

- `GET /health` — health check.
- `POST /predict` — predicts the fare. Invalid values, including locations outside the configured NYC bounds, are rejected with HTTP 422.

Run locally:

```bash
uvicorn src.api.main:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vendorid":2,"rate_code":1,"Pickup_longitude":-73.95,"Pickup_latitude":40.75,"Dropoff_longitude":-73.98,"Dropoff_latitude":40.76,"Passenger_count":1,"Trip_distance":3.5,"Trip_type":1,"pickup_hour":14,"pickup_weekday":2}'
```

Example response:

```json
{"predicted_fare": 17.28, "currency": "USD"}
```

The response above is an example; the actual prediction depends on the loaded model.

## Tests

Run the API test suite:

```bash
pytest -v
```

The suite contains 7 tests covering the health endpoint, valid predictions, expected distance/fare behavior, and invalid vendor, distance, location, and missing-field inputs.

## Docker

The model artifact `models/best_model.joblib` must be available before building the image.

```bash
docker build -t green-taxi-api:local .
docker run --rm -p 8000:8000 green-taxi-api:local
```

## CI/CD and AWS deployment

GitHub Actions workflow `.github/workflows/ci.yml` runs on pushes to `main` and can also be started manually. It:

1. Installs the API dependencies and test tools.
2. Authenticates to AWS using GitHub Actions OIDC.
3. Pulls the model artifact from the configured DVC remote.
4. Runs the pytest suite.
5. Builds the Docker image and smoke-tests `/health` and `/predict`.
6. Pushes the image to Amazon ECR.
7. Updates the Amazon ECS task definition and deploys the service to **ECS Fargate**, waiting for service stability.

The ECS task definition sends container logs to the CloudWatch log group `/ecs/green-taxi-v4-api`.

The GitHub Actions deployment workflow has completed successfully in the repository. This confirms the pipeline completed at that run; it does not by itself guarantee that the live API is currently healthy.

## Project structure

```
data/         raw data pointer, interim and processed data (DVC)
models/       trained model artifacts (DVC)
reports/      metrics, best parameters, and model comparison
src/
  data/       dataset cleaning
  features/   feature engineering and train/test split
  models/     training, tuning, evaluation, prediction
  api/        FastAPI application
tests/        API tests
params.yaml   pipeline and validation settings
dvc.yaml      DVC pipeline definition
Dockerfile
ecs-task-definition.json
.github/workflows/ci.yml
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, DVC with S3 storage, FastAPI, pytest, Docker, GitHub Actions, AWS IAM OIDC, Amazon ECR, Amazon ECS Fargate, and Amazon CloudWatch Logs.

## Current scope

This repository implements batch model training/evaluation, a prediction API, containerization, CI tests, and deployment through GitHub Actions to ECS Fargate. Continuous training, automated model-promotion/rollback gates, and CloudWatch alarms/metrics dashboards are not part of the current implementation.
