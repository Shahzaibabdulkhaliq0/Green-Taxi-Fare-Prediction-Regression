# test_api.py
# Tests for the API. Run with: pytest

# TestClient calls the API without starting a real server
from fastapi.testclient import TestClient

# Our app
from src.api.main import app

# Create the test client
client = TestClient(app)

# A valid trip, same as the example in predict_model.py
valid_trip = {
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


# /health must say ok
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# A valid trip must return a positive fare
def test_predict_valid_trip():
    response = client.post("/predict", json=valid_trip)
    assert response.status_code == 200
    assert response.json()["predicted_fare"] > 0
    assert response.json()["currency"] == "USD"


# A longer trip must not cost less than a shorter trip
def test_longer_trip_costs_more():
    short_trip = dict(valid_trip, Trip_distance=1.0)
    long_trip = dict(valid_trip, Trip_distance=15.0)
    short_fare = client.post("/predict", json=short_trip).json()["predicted_fare"]
    long_fare = client.post("/predict", json=long_trip).json()["predicted_fare"]
    assert long_fare > short_fare


# A wrong vendor code must be rejected
def test_invalid_vendor():
    response = client.post("/predict", json=dict(valid_trip, vendorid=9))
    assert response.status_code == 422


# A negative distance must be rejected
def test_negative_distance():
    response = client.post("/predict", json=dict(valid_trip, Trip_distance=-5))
    assert response.status_code == 422


# A location outside NYC must be rejected
def test_location_outside_nyc():
    response = client.post("/predict", json=dict(valid_trip, Pickup_latitude=10.0))
    assert response.status_code == 422


# A missing field must be rejected
def test_missing_field():
    incomplete_trip = dict(valid_trip)
    del incomplete_trip["Passenger_count"]
    response = client.post("/predict", json=incomplete_trip)
    assert response.status_code == 422