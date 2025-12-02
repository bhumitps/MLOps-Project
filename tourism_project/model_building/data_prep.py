
"""Data preparation script for the Tourism package prediction project.

Steps:
1. Load the raw tourism.csv file from the local data folder.
2. Clean and prepare the dataset (drop unused columns, handle missing values).
3. Split the data into train and test sets.
4. Save the processed train/test splits to CSV files.
5. Upload the processed files to the Hugging Face dataset repo.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Constants
RAW_DATA_PATH = "tourism_project/data/tourism.csv"
PROCESSED_DIR = "tourism_project/processed_data"
TARGET_COL = "ProdTaken"
DATASET_REPO_ID = "bhumitps/MLops"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# 1. Load raw data
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Could not find input file at {RAW_DATA_PATH}. "
                            "Please ensure 'tourism.csv' is placed in tourism_project/data.")

df = pd.read_csv(RAW_DATA_PATH)

# 2. Basic cleaning
# Drop columns that are identifiers or not useful for prediction
cols_to_drop = ["Unnamed: 0", "CustomerID"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Drop rows where target is missing (if any)
df = df.dropna(subset=[TARGET_COL])

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Save processed data
X_train_path = os.path.join(PROCESSED_DIR, "Xtrain.csv")
X_test_path = os.path.join(PROCESSED_DIR, "Xtest.csv")
y_train_path = os.path.join(PROCESSED_DIR, "ytrain.csv")
y_test_path = os.path.join(PROCESSED_DIR, "ytest.csv")

X_train.to_csv(X_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_train.to_csv(y_train_path, index=False)
y_test.to_csv(y_test_path, index=False)

print("Processed data saved to:", PROCESSED_DIR)

# 5. Upload processed data to Hugging Face dataset repo
files_to_upload = [X_train_path, X_test_path, y_train_path, y_test_path]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )

print("Processed data uploaded to Hugging Face dataset:", DATASET_REPO_ID)
