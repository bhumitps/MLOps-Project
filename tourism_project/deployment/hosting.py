from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load the environment variables (useful if running locally)
# In CI/CD, the token is pulled from secrets.
load_dotenv() 

# Initialize HfApi with the token from the environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define project constants
FOLDER_PATH = "tourism_project/deployment" # The local folder containing your app.py, Dockerfile, requirements.txt
REPO_ID = "bhumitps/MLops"              # The target Hugging Face Space repository (from Details.txt)
REPO_TYPE = "space"                     # The type of repository (it is a Space)

print(f"Starting upload of deployment files from {FOLDER_PATH} to Hugging Face Space: {REPO_ID}")

try:
    api.upload_folder(
        folder_path=FOLDER_PATH, 
        repo_id=REPO_ID, 
        repo_type=REPO_TYPE, 
        path_in_repo="" # Uploads contents directly to the root of the Space
    )
    print("\nDeployment files successfully uploaded to Hugging Face Space.")
    print(f"Deployment URL: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"\nDeployment failed during upload: {e}")
    print("Please ensure your HF_TOKEN is correctly set up as an environment variable or secret.")
