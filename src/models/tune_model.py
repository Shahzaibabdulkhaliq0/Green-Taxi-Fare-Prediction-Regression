# tune_model.py
# Notebook Step 11: tune the hyperparameters of the best model only, then save the tuned model.

# Path builds file paths that work on any computer
from pathlib import Path

# json: saves the best parameters in a readable file
import json

# joblib: loads and saves models
import joblib

# pandas: load table data
import pandas as pd

# yaml: reads the settings file params.yaml
import yaml

# sys
import sys

# RandomizedSearchCV: tries random combinations of settings and keeps the best one
from sklearn.model_selection import RandomizedSearchCV

# MLFlow : Start
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.mlflow_utils import log_tuned_model
# MLFLow : END

# Read the settings from params.yaml (nothing is hardcoded in this script)
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)["tune_model"]

# Folders for the data, the saved models and the reports
processed_folder = PROJECT_ROOT / "data" / "processed"
models_folder = PROJECT_ROOT / "models"
reports_folder = PROJECT_ROOT / "reports"

# Load the training data
train = pd.read_csv(processed_folder / "train.csv")
X_train = train.drop(columns=[params["target"]])
y_train = train[params["target"]]

# Read which model won in train_model.py
best_model_name = (models_folder / "best_model_name.txt").read_text()
print("Tuning:", best_model_name)

# Load the untrained best pipeline (preprocessing + model settings) saved by train_model.py
pipeline = joblib.load(models_folder / "untrained_pipeline.joblib")

# Pick the settings to try for the winning model (params: param_grids; "Model__" means: a setting of the Model step inside the pipeline)
param_dist = params["param_grids"][best_model_name]

# Randomized search with cross validation (params: n_iter = number of random combinations, cv_folds); the best score is the lowest RMSE (error in dollars)
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=params["n_iter"],
    cv=params["cv_folds"],
    scoring="neg_root_mean_squared_error",
    random_state=params["random_state"],
    n_jobs=1,
    verbose=2
)

# Run the search on the training data
random_search.fit(X_train, y_train)

# The best model found (it is already trained again on the complete training data)
best_model = random_search.best_estimator_

# Show the best settings and the best cross validation error
print("Best Parameters:", random_search.best_params_)
print("Best CV RMSE   :", -random_search.best_score_)

# Create the output folders if they do not exist yet
models_folder.mkdir(parents=True, exist_ok=True)
reports_folder.mkdir(parents=True, exist_ok=True)

# Save the tuned model: this is the final model used by evaluate_model.py and predict_model.py
joblib.dump(best_model, models_folder / "best_model.joblib")

# Save the best settings in a readable file
with open(reports_folder / "best_params.json", "w") as file:
    json.dump({"model": best_model_name, "best_params": random_search.best_params_, "best_cv_rmse": -random_search.best_score_}, file, indent=4, default=str)

# MLFlow Start From Here
registered_version = log_tuned_model(
    best_model=best_model,
    best_model_name=best_model_name,
    best_params=random_search.best_params_,
    best_cv_rmse=-random_search.best_score_,
    best_params_file=reports_folder / "best_params.json"
)

print("Registered MLflow model version:", registered_version)
# Mlflow END Here

print("Saved the tuned model in:", models_folder)