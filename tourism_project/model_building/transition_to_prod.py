import os
import mlflow
from mlflow.tracking import MlflowClient

# Read env vars – pipeline passes MLFLOW_TRACKING_URI and MODEL_NAME
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MODEL_NAME = os.getenv("MODEL_NAME", "Tourism_Purchase_Predictor")

def main():
    # Point MLflow to the tracking backend
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    print(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Promoting model: {MODEL_NAME}")

    try:
        # Get all versions of the model
        versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
        if not versions:
            raise RuntimeError(f"No versions found for model '{MODEL_NAME}'")

        # Pick the latest version number
        latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
        print(f"Latest version found: {latest_version.version}")

        # Move that version to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print(
            f"Model '{MODEL_NAME}' version {latest_version.version} "
            f"has been transitioned to 'Production'."
        )

    except Exception as e:
        print(f"Could not transition model to Production. Error: {e}")
        raise

if __name__ == "__main__":
    main()
