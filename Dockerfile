# Base image: small Python 3.14 (same Python as our local environment)
FROM python:3.14-slim

# libgomp1: xgboost needs this system library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Work inside /app
WORKDIR /app

# Install packages first (Docker caches this step, so rebuilds are fast)
# The cache mount keeps downloaded files, so a failed build does not start from zero
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries=10 --resume-retries=10 -r requirements-api.txt

# Copy only what the API needs
COPY src/__init__.py src/__init__.py
COPY src/api/ src/api/
COPY params.yaml .
COPY models/best_model.joblib models/best_model.joblib

# The API listens on port 8000
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
