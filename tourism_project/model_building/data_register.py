# for data manipulation
import pandas as pd
# for os functions
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for file path and repository details
# Reads the file from the project data directory
FILE_PATH = "tourism_project/data/tourism.csv" 
REPO_ID = "bhumitps/amlops" 
REPO_TYPE = "dataset"

# Check if the file exists before proceeding
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Source file not found at: {FILE_PATH}")

# Initialize HfApi with the token from the environment variable
api = HfApi(token=os.getenv("HF_TOKEN")) 

# Upload the file
print(f"Uploading {FILE_PATH} to {REPO_ID}...")

# THIS IS THE CRITICAL FIX: Use upload_file for a single file.
api.upload_file( 
    path_or_fileobj=FILE_PATH,
    path_in_repo=FILE_PATH.split("/")[-1], # tourism.csv
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

print(f"Dataset successfully registered to Hugging Face Hub: hf://datasets/{REPO_ID}/tourism.csv")
