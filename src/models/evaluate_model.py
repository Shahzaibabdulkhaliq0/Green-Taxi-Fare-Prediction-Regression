# evaluate_model.py
# Check the final (tuned) model on the test data, which the model has never seen, and save the scores.

# Path builds file paths that work on any computer
from pathlib import Path

# json: saves the scores in a readable file
import json

# joblib: loads the saved model
import joblib

# numpy and pandas: numbers and table data
import numpy as np
import pandas as pd

# yaml: reads the settings file params.yaml
import yaml

# Error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# The project folder is two folders above this file
project_root = Path(__file__).resolve().parents[2]

# Read the settings from params.yaml
with open(project_root / "params.yaml") as file:
    params = yaml.safe_load(file)["train_model"]

# Folders for the data, the saved models and the reports
processed_folder = project_root / "data" / "processed"
models_folder = project_root / "models"
reports_folder = project_root / "reports"

# Load the test data and split it into inputs (X) and the target (y)
test = pd.read_csv(processed_folder / "test.csv")
X_test = test.drop(columns=[params["target"]])
y_test = test[params["target"]]

# Load the final tuned model
model = joblib.load(models_folder / "best_model.joblib")

# Predict the fares of the test data
y_pred = model.predict(X_test)

# MAE: the average mistake in dollars. RMSE: like MAE but big mistakes count more. R2: 1.0 is perfect, 0 is no better than the average.
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Show the scores
print("Test MAE :", round(mae, 4))
print("Test RMSE:", round(rmse, 4))
print("Test R2  :", round(r2, 4))

# Show the first 10 real fares next to the predicted fares, to see the model's mistakes with your own eyes
comparison = pd.DataFrame({"Actual Fare": y_test.values[:10], "Predicted Fare": y_pred[:10].round(2)})
print(comparison)

# Create the reports folder if it does not exist yet
reports_folder.mkdir(parents=True, exist_ok=True)

# Save the scores
with open(reports_folder / "metrics.json", "w") as file:
    json.dump({"test_mae": mae, "test_rmse": rmse, "test_r2": r2}, file, indent=4)
print("Saved:", reports_folder / "metrics.json")
