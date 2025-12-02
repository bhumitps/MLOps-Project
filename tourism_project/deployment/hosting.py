from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

# Load env vars when running locally (.env). In CI/CD, secrets are injected directly.
load_dotenv()

# ---- Hugging Face config ----
REPO_ID = "bhumitps/amlops"   # Space repo
REPO_TYPE = "space"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "Set it in your .env for local runs or as a GitHub Secret in CI."
    )

# Initialize API client
api = HfApi(token=HF_TOKEN)

# Ensure the Space exists (create if missing)
create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    exist_ok=True,     # won't fail if the Space already exists
)

# Local folder containing app.py, Dockerfile, requirements.txt, etc.
FOLDER_PATH = "tourism_project/deployment"

print(f"Starting upload of deployment files from {FOLDER_PATH} to Hugging Face Space: {REPO_ID}")

try:
    api.upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="",   # upload to root of the Space
        # Optional: ignore build artefacts
        # ignore_patterns=["__pycache__/", "*.pyc", ".DS_Store"]
    )
    print("\nDeployment files successfully uploaded to Hugging Face Space.")
    print(f"Deployment URL: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"\nDeployment failed during upload: {e}")
    print("Please ensure your HF_TOKEN is correctly set and has write access to the Space.")
