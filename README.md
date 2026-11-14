# Green Taxi Fare Prediction — MLOps

An end-to-end machine learning project that predicts NYC green taxi fares using a reproducible ML pipeline, a FastAPI prediction service, Docker, GitHub Actions, and AWS.

**Dataset:** NYC TLC 2015 Green Taxi Trip Data  
**Target:** `Fare_amount`  
**Deployment:** Amazon ECS Fargate  
**AWS Region:** `eu-north-1`

## 1. Project Architecture

The project has two separate workflows:

### ML development and training

Raw NYC Taxi Data
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Train / Test Split
       |
       v
Model Comparison
       |
       v
Hyperparameter Tuning
       |
       v
Model Evaluation
       |
       v
DVC Artifacts + S3

### API CI/CD and deployment

GitHub Push to main
       |
       v
GitHub Actions
       |
       v
AWS Authentication (OIDC)
       |
       v
Pull Model from DVC / S3
       |
       v
Pytest
       |
       v
Docker Build
       |
       v
Container Smoke Tests
       |
       v
Push Image to Amazon ECR
       |
       v
Deploy to Amazon ECS Fargate
       |
       v
CloudWatch Logs and Alarms

Model training and API deployment are separate processes. The current CI/CD workflow deploys the existing DVC-tracked model; it does not automatically retrain the model.

## 2. Machine Learning Pipeline

The training pipeline is orchestrated using DVC.

| Stage | Script | Output |
|---|---|---|
| Data cleaning | `src/data/make_dataset.py` | `data/interim/cleaned_data.csv` |
| Feature engineering | `src/features/build_features.py` | Train/test datasets |
| Model comparison | `src/models/train_model.py` | Model comparison and selected pipeline |
| Hyperparameter tuning | `src/models/tune_model.py` | `models/best_model.joblib` |
| Model evaluation | `src/models/evaluate_model.py` | `reports/metrics.json` |

### Dataset preparation

- 50,000 cleaned records are sampled.
- Random seed: `42`.
- Train/test split: 80/20.
- Target variable: `Fare_amount`.
- Preprocessing is included in the scikit-learn pipeline.
- Numerical and categorical features are handled separately.

### Model comparison

Models are compared using 5-fold cross-validation on training data.

| Model | CV RMSE Mean | CV RMSE Std |
|---|---:|---:|
| Random Forest | 2.5064 | 0.2470 |
| XGBoost | 2.6075 | 0.2977 |
| Ridge Regression | 2.7559 | 0.2901 |

Random Forest was selected using cross-validation RMSE, then tuned using `RandomizedSearchCV`.

### Final recorded evaluation

The committed evaluation report contains:

| Metric | Result |
|---|---:|
| Test MAE | 1.2529 |
| Test RMSE | 2.5205 |
| Test R² | 0.9130 |

The test dataset is reserved for final evaluation and is not used for model selection.

### Feature and leakage control

The model uses 11 input features:

- Vendor and rate code
- Pickup and dropoff coordinates
- Passenger count
- Trip distance
- Trip type
- Pickup hour
- Pickup weekday

Fields that are unavailable at prediction time or derived from the fare are excluded.

## 3. Data and Model Versioning — DVC

DVC tracks the raw dataset, pipeline outputs, and trained model artifacts.

**Remote storage:**

- S3 bucket: `green-taxi-v4-dvc-010928224571`
- Prefix: `dvc/`
- Region: `eu-north-1`

The DVC configuration is stored in `.dvc/config`. AWS credentials are required to access the remote.

### Reproduce the pipeline

Install the project dependencies and configure AWS access first.

pip install -r requirements.txt
pip install mlflow
dvc pull
dvc repro

The separate API dependency file is `requirements-api.txt`.

## 4. Experiment Tracking — MLflow

MLflow tracking integration is implemented in:

- `src/tracking/mlflow_utils.py`
- `src/models/train_model.py`
- `src/models/tune_model.py`
- `src/models/evaluate_model.py`

Current design:

- SQLite tracking backend: `mlflow.db`
- Local artifact directory: `mlruns/`
- Experiment: `green-taxi-fare-prediction`
- Registered model: `GreenTaxiFarePrediction`

The local database and artifacts are excluded from Git.

**Current limitation:** MLflow is not part of the GitHub Actions training workflow. Its local tracking and registry implementation is not yet a production-hosted model registry or an automated model approval/promotion process.

## 5. Prediction API

The FastAPI application is located at:

`src/api/main.py`

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict taxi fare |

The API validates incoming trip data using Pydantic and configured NYC ranges.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vendorid":2,"rate_code":1,"Pickup_longitude":-73.95,"Pickup_latitude":40.75,"Dropoff_longitude":-73.98,"Dropoff_latitude":40.76,"Passenger_count":1,"Trip_distance":3.5,"Trip_type":1,"pickup_hour":14,"pickup_weekday":2}'
```

Example response:

```json
{
  "predicted_fare": 17.28,
  "currency": "USD"
}
```

The value is illustrative; actual predictions depend on the loaded model.

## 6. Testing

The API test suite is located in `tests/test_api.py`.

It currently contains 7 tests covering:

- Health endpoint
- Valid prediction
- Longer-trip prediction behavior
- Invalid vendor
- Negative distance
- Invalid NYC location
- Missing input field

Run tests:

```bash
pytest -v
```

The current CI workflow executes these API tests. It does not execute the full DVC training pipeline or dedicated data/model validation tests.

## 7. Docker

The Docker image contains the API, its runtime dependencies, configuration, and the DVC-pulled model artifact.

Build:

```bash
docker build -t green-taxi-api:local .
```

Run:

```bash
docker run --rm -p 8000:8000 green-taxi-api:local
```

The model artifact must be available at `models/best_model.joblib` before building.

## 8. CI/CD — GitHub Actions

Workflow:

`.github/workflows/ci.yml`

Triggers:

- Push to `main`
- Manual `workflow_dispatch`

The workflow performs:

1. Checkout repository.
2. Set up Python 3.14.
3. Install API, testing, and DVC dependencies.
4. Authenticate to AWS using GitHub OIDC.
5. Pull the model from DVC/S3.
6. Run pytest.
7. Authenticate to Amazon ECR.
8. Build Docker image with a unique Git SHA and run tag.
9. Smoke-test `/health` and `/predict`.
10. Push image to ECR.
11. Render ECS task definition.
12. Deploy to ECS and wait for service stability.

### Latest verified CI run

- Run ID: `43572801102`
- Result: Success
- Tests: Success
- Docker build: Success
- Container smoke test: Success
- ECR push: Success
- ECS deployment: Success

## 9. AWS Deployment Infrastructure

The active V4 deployment uses:

| AWS resource | Current configuration |
|---|---|
| Region | `eu-north-1` |
| S3 bucket | `green-taxi-v4-dvc-010928224571` |
| ECR repository | `green-taxi-v4-api` |
| ECS cluster | `green-taxi-v4-cluster` |
| ECS service | `green-taxi-v4-api-service` |
| Launch type | Fargate |
| Task definition | `green-taxi-v4-api:9` |
| Task CPU | 1 vCPU |
| Task memory | 4 GB |
| Desired tasks | 1 |
| Running tasks | 1 |
| CloudWatch log group | `/ecs/green-taxi-v4-api` |

### IAM and authentication

GitHub Actions authenticates through AWS IAM OIDC.

Role:

`green-taxi-v4-github-actions`

The role has scoped permissions for the V4 DVC bucket, ECR repository, and ECS deployment resources.

The ECS task execution role handles ECR image pulls and CloudWatch log delivery.

### Deployment safety

- ECS rolling deployment.
- ECS deployment circuit breaker enabled.
- Automatic ECS deployment rollback enabled.
- Container health check configured.

The ECS rollback protects service deployments. It is not a model-quality-based rollback mechanism.

## 10. Monitoring and Logging

Current AWS monitoring includes:

- CloudWatch container logs.
- CPU utilization alarm.
- Memory utilization alarm.
- SNS email notifications.

CloudWatch log retention is configured for 7 days.

Current monitoring does not include:

- Model prediction quality monitoring.
- Data drift or concept drift detection.
- Prediction latency and request-volume dashboards.
- Automated retraining triggers.
- Model-quality-based rollback.

## 11. Continuous Training — Current Scope

**Continuous Training (CT) is intentionally removed from the current repository.**

There is no active CT workflow in `.github/workflows/`.

The historical GitHub Actions CT runs remain visible in Actions history, but they are not active workflows.

Model training is currently a manually executed, reproducible DVC pipeline. New models are not automatically promoted or deployed.

## 12. Remaining Engineering Work

The following items are not claimed as implemented:

1. Add and pin MLflow in training dependencies.
2. Make MLflow tracking reproducible and decide on durable tracking storage.
3. Add data schema and data-quality validation.
4. Add dedicated unit tests for feature engineering and model training.
5. Add model validation and an explicit approval/promotion process.
6. Add model performance and data drift monitoring.
7. Add a tested model rollback and recovery procedure.
8. Add API latency/request metrics and operational dashboards.
9. Improve deployment security with HTTPS and an appropriate public access architecture.

Continuous Training is not part of the current scope.

## 13. Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .dvc/
│   └── config
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── tracking/
├── tests/
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── requirements-api.txt
└── ecs-task-definition.json
```

## 14. Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- DVC
- MLflow (local tracking implementation)
- FastAPI
- Pydantic
- Pytest
- Docker
- GitHub Actions
- AWS IAM OIDC
- Amazon S3
- Amazon ECR
- Amazon ECS Fargate
- Amazon CloudWatch
- Amazon SNS

## 15. Reference

Google Cloud — [MLOps: Continuous delivery and automation pipelines in machine learning](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning).

This project implements selected MLOps practices. It does not claim complete Google MLOps Level 1 or Level 2 maturity.