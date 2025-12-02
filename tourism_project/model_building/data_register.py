import os
from huggingface_hub import HfApi, create_repo

# ---- Config ----
FILE_PATH = "tourism_project/data/tourism.csv"   # local CSV
REPO_ID = os.getenv("REPO_ID", "bhumitps/amlops")  # dataset repo id (NOT the space type)
REPO_TYPE = "dataset"

# Check local file exists
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Source file not found at: {FILE_PATH}")

# Get token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "Set it as a GitHub Secret and in your local env if running locally."
    )

# Init API client
api = HfApi(token=HF_TOKEN)

# 🔹 Critical: create the DATASET repo if it doesn't exist
create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,   # <--- must be "dataset"
    private=False,
    exist_ok=True,         # don't fail if it already exists
    token=HF_TOKEN,
)

print(f"Uploading {FILE_PATH} to dataset repo {REPO_ID}...")

# Upload CSV into a nice path inside the dataset
api.upload_file(
    path_or_fileobj=FILE_PATH,
    path_in_repo="data/tourism.csv",  # path inside the dataset repo
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

print(f"Dataset successfully registered at: hf://datasets/{REPO_ID}/data/tourism.csv")
