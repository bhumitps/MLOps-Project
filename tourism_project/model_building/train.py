
"""Model training script for Tourism package prediction.

- Loads processed train/test splits from tourism_project/processed_data.
- Builds a preprocessing + XGBoost classification pipeline.
- Trains the model and evaluates it.
- Logs metrics to MLflow (if the tracking server is running).
- Saves the best model locally and uploads it to Hugging Face Hub as a model repo.
"""

import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import xgboost as xgb
import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Paths
PROCESSED_DIR = "tourism_project/processed_data"
MODEL_DIR = "tourism_project/model_building"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# HF model repo details
HF_MODEL_REPO_ID = "bhumitps/MLops"
HF_MODEL_REPO_TYPE = "model"

# Initialize HF API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load processed data
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "Xtrain.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "Xtest.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "ytrain.csv"), squeeze=True)
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "ytest.csv"), squeeze=True)

# If y_* are saved as single-column dataframes, convert to Series
if not isinstance(y_train, pd.Series):
    y_train = y_train.iloc[:, 0]
if not isinstance(y_test, pd.Series):
    y_test = y_test.iloc[:, 0]

# Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# XGBoost classifier
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ]
)

# Configure MLflow (assumes an MLflow server is available at this URI)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-experiment")

with mlflow.start_run(run_name="xgboost_tourism"):
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print("Accuracy:", accuracy)
    print("ROC-AUC:", auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Log params and metrics to MLflow
    mlflow.log_params(
        {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
    )
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", auc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

# Save model locally
joblib.dump(model, MODEL_PATH)
print(f"Best tourism model saved locally at: {{MODEL_PATH}}")

# Ensure HF model repo exists
try:
    api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type=HF_MODEL_REPO_TYPE)
    print(f"Model repo '{{HF_MODEL_REPO_ID}}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{{HF_MODEL_REPO_ID}}' not found. Creating new model repo...")
    create_repo(repo_id=HF_MODEL_REPO_ID, repo_type=HF_MODEL_REPO_TYPE, private=False)
    print(f"Model repo '{{HF_MODEL_REPO_ID}}' created.")

# Upload model file to HF Hub
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo=os.path.basename(MODEL_PATH),
    repo_id=HF_MODEL_REPO_ID,
    repo_type=HF_MODEL_REPO_TYPE,
)

print("Best tourism model uploaded to Hugging Face Hub model repo:", HF_MODEL_REPO_ID)
