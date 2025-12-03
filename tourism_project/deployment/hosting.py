from huggingface_hub import HfApi, create_repo
import os

# Optional local .env loading (safe in CI)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

# Ensure the Space exists as a *Docker Space*
create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    space_sdk="docker",
    exist_ok=True,
)

# Local folder containing Dockerfile, app, requirements, etc.
FOLDER_PATH = "tourism_project/deployment"

print(f"Starting upload of deployment files from {FOLDER_PATH} to Hugging Face Space: {REPO_ID}")

try:
    api.upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="",   # upload to root of the Space
    )
    print("\nDeployment files successfully uploaded to Hugging Face Space.")
    print(f"Live at: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"\nDeployment failed during upload: {e}")
