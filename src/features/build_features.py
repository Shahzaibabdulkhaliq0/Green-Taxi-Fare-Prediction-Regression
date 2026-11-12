# build_features.py
# Notebook Steps 6 to 8: create the date features, choose X and y, split into train and test, save both.

# Path builds file paths that work on any computer
from pathlib import Path

# pandas: load and change table data
import pandas as pd

# yaml: reads the settings file params.yaml
import yaml

# train_test_split: splits our data into a training part and a testing part
from sklearn.model_selection import train_test_split

# The project folder is two folders above this file
project_root = Path(__file__).resolve().parents[2]

# Read the settings from params.yaml (nothing is hardcoded in this script)
with open(project_root / "params.yaml") as file:
    params = yaml.safe_load(file)["build_features"]

# Where the cleaned data was saved by make_dataset.py
clean_path = project_root / "data" / "interim" / "cleaned_data.csv"

# Where the train and test files will be saved
processed_folder = project_root / "data" / "processed"

# Load the cleaned data
df = pd.read_csv(clean_path)

# Convert the text date (like 02/10/2015 08:46:15 AM) into a real date-time; unreadable dates become NaT (missing)
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], format=params["pickup_datetime_format"], errors="coerce")

# Remove rows whose date could not be read
df = df.dropna(subset=["pickup_datetime"])

# Create pickup_hour: the hour of the day (0 to 23)
df["pickup_hour"] = df["pickup_datetime"].dt.hour

# Create pickup_weekday: the day of the week (0 = Monday ... 6 = Sunday)
df["pickup_weekday"] = df["pickup_datetime"].dt.day_of_week

# The columns we keep as inputs (params: features). We list the columns to KEEP, so the target can never slip into X by mistake.
features = params["features"]

# X is the table of input columns
X = df[features]

# y is the target column: the fare we want to predict (params: target)
y = df[params["target"]]

# Split the data into train and test (params: test_size, random_state). The same random_state gives the same split every time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"])

# Join inputs and target again, so each file (train / test) holds everything in one table
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Check the size of each part
print("Train Shape :", train.shape)
print("Test  Shape :", test.shape)

# Create the output folder if it does not exist yet
processed_folder.mkdir(parents=True, exist_ok=True)

# Save the train and test data
train.to_csv(processed_folder / "train.csv", index=False)
test.to_csv(processed_folder / "test.csv", index=False)
print("Saved train.csv and test.csv in:", processed_folder)