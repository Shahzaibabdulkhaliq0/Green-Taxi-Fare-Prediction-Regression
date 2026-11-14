
# train_model.py
# Compare models using cross-validation on training data only.
# The test dataset is reserved for final model evaluation.

from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.base import clone

# Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Cross-validation
from sklearn.model_selection import cross_val_score

# MLFlow : Start
import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.mlflow_utils import log_model_comparison
# MLflow : END

# Load configuration
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)["train_model"]

# Project folders
processed_folder = PROJECT_ROOT / "data" / "processed"
models_folder = PROJECT_ROOT / "models"
reports_folder = PROJECT_ROOT / "reports"

# Load ONLY training data.
# test.csv is reserved for final evaluation.
train = pd.read_csv(processed_folder / "train.csv")

# Split training data into features and target
X_train = train.drop(columns=[params["target"]])
y_train = train[params["target"]]

# Numerical and categorical columns
numerical_columns = [
    "Pickup_longitude",
    "Pickup_latitude",
    "Dropoff_longitude",
    "Dropoff_latitude",
    "Passenger_count",
    "Trip_distance",
    "pickup_hour",
    "pickup_weekday"
]

categorical_columns = [
    "vendorid",
    "rate_code",
    "Trip_type"
]

# Numerical preprocessing
numerical_pipeline = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="median")),
    ("Scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("Encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="if_binary",
        sparse_output=False
    ))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("Numerical Pipeline", numerical_pipeline, numerical_columns),
    ("Categorical Pipeline", categorical_pipeline, categorical_columns)
])

# Ridge Regression
ridge_pipeline = Pipeline(steps=[
    ("Preprocessor", preprocessor),
    ("Model", Ridge())
])

# Model 2: Random Forest (many decision trees, results are averaged). max_depth and min_samples_leaf keep the trees small, so the saved file stays small.
rf_pipeline = Pipeline(steps=[
    ("Preprocessor", preprocessor),
    ("Model", RandomForestRegressor(n_estimators=params["rf_n_estimators"], max_depth=params["rf_max_depth"], min_samples_leaf=params["rf_min_samples_leaf"], random_state=params["random_state"], n_jobs=-1))
])

# Model 3: XGBoost (decision trees built one after another, each one fixes the errors of the previous ones)
xgb_pipeline = Pipeline(steps=[
    ("Preprocessor", preprocessor),
    ("Model", XGBRegressor(n_estimators=params["xgb_n_estimators"], random_state=params["random_state"], n_jobs=-1))
])

# Models to compare
models = {
    "Ridge Regression": ridge_pipeline,
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline
}

# Store cross-validation results
results = []

# Compare models using training data only
for name, model in models.items():

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=params["cv_folds"],
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )

    results.append({
        "Model": name,
        "CV RMSE Mean": -cv_scores.mean(),
        "CV RMSE Std": cv_scores.std()
    })

# Sort models by cross-validation RMSE
comparison_df = pd.DataFrame(results).sort_values(
    by="CV RMSE Mean",
    ascending=True
)

print(comparison_df.to_string(index=False))

# Select the best model using cross-validation
best_model_name = comparison_df["Model"].iloc[0]

print("Best model:", best_model_name)

# Create output folders
models_folder.mkdir(parents=True, exist_ok=True)
reports_folder.mkdir(parents=True, exist_ok=True)

# Save model comparison
comparison_df.to_csv(
    reports_folder / "model_comparison.csv",
    index=False
)

## Start MLFlow From Here

log_model_comparison(
    results=results,
    selected_model=best_model_name,
    cv_folds=params["cv_folds"]
)

## MLFlow END Here

# Save selected model name
(models_folder / "best_model_name.txt").write_text(best_model_name)

# Save an untrained copy of the selected pipeline for tuning
joblib.dump(
    clone(models[best_model_name]),
    models_folder / "untrained_pipeline.joblib"
)

print(
    "Saved the best model name and untrained pipeline in:",
    models_folder
)