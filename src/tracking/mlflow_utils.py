
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_PATH = PROJECT_ROOT / "mlruns"

# Local SQLite tracking backend
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

# Experiment and registered model names
EXPERIMENT_NAME = "green-taxi-fare-prediction"
REGISTERED_MODEL_NAME = "GreenTaxiFarePrediction"

def configure_mlflow():
    """Configure local MLflow tracking and model registry."""

    MLFLOW_ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_PATH.resolve().as_uri()
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    return mlflow, client

def log_model_comparison(results, selected_model, cv_folds):
    """Log model comparison results."""

    mlflow, _ = configure_mlflow()

    with mlflow.start_run(run_name="model_comparison"):

        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("selected_model", selected_model)

        for result in results:
            model_key = (
                result["Model"]
                .lower()
                .replace(" ", "_")
            )

            mlflow.log_metrics({
                f"{model_key}_cv_rmse_mean":
                    float(result["CV RMSE Mean"]),

                f"{model_key}_cv_rmse_std":
                    float(result["CV RMSE Std"])
            })

        comparison_file = (
            PROJECT_ROOT / "reports" / "model_comparison.csv"
        )

        if comparison_file.exists():
            mlflow.log_artifact(str(comparison_file))

def log_tuned_model(
    best_model,
    best_model_name,
    best_params,
    best_cv_rmse,
    best_params_file
):
    """Log tuned model and register a new model version."""

    mlflow, _ = configure_mlflow()

    with mlflow.start_run(run_name="hyperparameter_tuning") as run:

        mlflow.log_param("selected_model", best_model_name)

        mlflow.log_params({
            str(key): str(value)
            for key, value in best_params.items()
        })

        mlflow.log_metric(
            "best_cv_rmse",
            float(best_cv_rmse)
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
            ]
        )

        if best_params_file.exists():
            mlflow.log_artifact(str(best_params_file))

        model_uri = f"runs:/{run.info.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    return registered_model.version

def log_evaluation_metrics(metrics, metrics_file):
    """Log final test evaluation metrics."""

    mlflow, _ = configure_mlflow()

    with mlflow.start_run(run_name="final_test_evaluation"):

        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
        })

        if metrics_file.exists():
            mlflow.log_artifact(str(metrics_file))