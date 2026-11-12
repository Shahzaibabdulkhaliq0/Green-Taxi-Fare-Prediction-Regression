# main.py
# FastAPI app: takes trip details and returns the predicted fare.

# Path builds file paths that work on any computer
from pathlib import Path

# joblib: loads the saved model
import joblib

# pandas: makes the one-row table the model needs
import pandas as pd

# yaml: reads params.yaml
import yaml

# FastAPI: the web framework. BaseModel and Field: check the input values
from fastapi import FastAPI
from pydantic import BaseModel, Field

# The project folder is two folders above this file
project_root = Path(__file__).resolve().parents[2]

# Read the valid ranges from params.yaml (same limits used in cleaning)
with open(project_root / "params.yaml") as f:
    limits = yaml.safe_load(f)["make_dataset"]

# Load the final tuned model once, when the app starts
model = joblib.load(project_root / "models" / "best_model.joblib")

# Create the app
app = FastAPI(title="Green Taxi Fare Prediction API")


# The input of one trip. Wrong values are rejected automatically (error 422).
class Trip(BaseModel):
    vendorid: int = Field(ge=min(limits["vendor_codes"]), le=max(limits["vendor_codes"]))
    rate_code: int = Field(ge=limits["rate_code_min"], le=limits["rate_code_max"])
    Pickup_longitude: float = Field(ge=limits["longitude_min"], le=limits["longitude_max"])
    Pickup_latitude: float = Field(ge=limits["latitude_min"], le=limits["latitude_max"])
    Dropoff_longitude: float = Field(ge=limits["longitude_min"], le=limits["longitude_max"])
    Dropoff_latitude: float = Field(ge=limits["latitude_min"], le=limits["latitude_max"])
    Passenger_count: int = Field(ge=limits["passenger_min"], le=limits["passenger_max"])
    Trip_distance: float = Field(ge=limits["distance_min"], le=limits["distance_max"])
    Trip_type: int = Field(ge=1, le=2)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_weekday: int = Field(ge=0, le=6)


# Health check: tells us the API is running
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction: trip in, fare out
@app.post("/predict")
def predict(trip: Trip):
    # One-row table. Column order is the same as in training.
    trip_df = pd.DataFrame([trip.model_dump()])
    predicted_fare = model.predict(trip_df)[0]
    return {"predicted_fare": round(float(predicted_fare), 2), "currency": "USD"}