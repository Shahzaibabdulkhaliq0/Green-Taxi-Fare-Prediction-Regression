# Path builds file paths that work on any computer
from pathlib import Path

# pandas: load and clean table data
import pandas as pd

# yaml: reads the settings file params.yaml
import yaml

# The project folder is two folders above this file (src/data/make_dataset.py -> project folder)
project_root = Path(__file__).resolve().parents[2]

# Read the settings from params.yaml (the cleaning rules are NOT hardcoded in this script)
with open(project_root / "params.yaml") as file:
    params = yaml.safe_load(file)["make_dataset"]

# Where the raw data is stored
raw_path = project_root / "data" / "raw" / "2015_Green_Taxi_Trip_Data.csv"

# Where the cleaned data will be saved
clean_path = project_root / "data" / "interim" / "cleaned_data.csv"

# Load the raw CSV file. low_memory=False makes pandas read the whole file at once, so column types are chosen correctly.
df = pd.read_csv(raw_path, low_memory=False)

# Convert 'Pickup_longitude' & 'vendorid' to 'float64' from 'str' 
df["Pickup_longitude"] = pd.to_numeric(df["Pickup_longitude"], errors="coerce")
df["vendorid"] = pd.to_numeric(df["vendorid"], errors="coerce")

# Remove Duplicates Rows
df = df.drop_duplicates()

# Keep only valid vendor codes (params: vendor_codes). Rows with an unreadable vendorid (NaN) are removed here too.
df = df[df["vendorid"].isin(params["vendor_codes"])]

# Keep fares between fare_min and fare_max (removes errors like $2,084 but keeps very long trips)
df = df[df["Fare_amount"].between(params["fare_min"], params["fare_max"])]

# Keep trips between distance_min and distance_max miles (a 0-mile trip has no real distance, 374 miles is an error)
df = df[df["Trip_distance"].between(params["distance_min"], params["distance_max"])]

# Keep trips with a valid number of passengers (a trip needs at least one passenger)
df = df[df["Passenger_count"].between(params["passenger_min"], params["passenger_max"])]

# Keep valid rate codes (data dictionary: 1 to 6, 99 means unknown)
df = df[df["rate_code"].between(params["rate_code_min"], params["rate_code_max"])]

# Keep trips whose pickup and dropoff latitude is inside the NYC area (this removes the (0, 0) points)
df = df[df["Pickup_latitude"].between(params["latitude_min"], params["latitude_max"]) & df["Dropoff_latitude"].between(params["latitude_min"], params["latitude_max"])]

# Keep trips whose pickup and dropoff longitude is inside the NYC area (this includes the airports)
df = df[df["Pickup_longitude"].between(params["longitude_min"], params["longitude_max"]) & df["Dropoff_longitude"].between(params["longitude_min"], params["longitude_max"])]

# Take a random sample of the cleaned rows (params: sample_size). random_state makes the SAME rows come every run.
df = df.sample(n=params["sample_size"], random_state=params["random_state"])
print("After random sample         :", len(df))

# Create output directory
clean_path.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned data
df.to_csv(clean_path, index=False)

print("All steps executed successfully!")