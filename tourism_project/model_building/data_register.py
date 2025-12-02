
"""Register the local tourism dataset on Hugging Face Hub as a dataset repo.

This script:
- Ensures the dataset repo exists (creates it if needed).
- Uploads the local 'tourism_project/data' folder (containing tourism.csv) to the repo.
"""

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "bhumitps/MLops"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{{repo_id}}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{{repo_id}}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{{repo_id}}' created.")

# Step 2: Upload local data folder to Hugging Face dataset
api.upload_folder(
    folder_path="tourism_project/data",  # should contain tourism.csv
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Local tourism data successfully registered on Hugging Face!")
