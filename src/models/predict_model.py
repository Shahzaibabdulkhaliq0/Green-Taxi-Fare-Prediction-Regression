# predict_model.py
# Predict the fare of ONE new trip with the final model. Change the values below and run the script again.

# Path builds file paths that work on any computer
from pathlib import Path

# joblib: loads the saved model
import joblib

# pandas: makes the one-row table the model needs
import pandas as pd

# The project folder is two folders above this file
project_root = Path(__file__).resolve().parents[2]

# Load the final tuned model
model = joblib.load(project_root / "models" / "best_model.joblib")

# The new trip. These 11 inputs are exactly the columns the model was trained on.
trip = {
    "vendorid": 2,
    "rate_code": 1,
    "Pickup_longitude": -73.95,
    "Pickup_latitude": 40.75,
    "Dropoff_longitude": -73.98,
    "Dropoff_latitude": 40.76,
    "Passenger_count": 1,
    "Trip_distance": 3.5,
    "Trip_type": 1,
    "pickup_hour": 14,
    "pickup_weekday": 2,
}

# Put the trip in a one-row table (the model expects a table, not a dictionary)
trip_df = pd.DataFrame([trip])

# Predict the fare
predicted_fare = model.predict(trip_df)[0]

# Show the result
print("Predicted fare: $", round(float(predicted_fare), 2))
