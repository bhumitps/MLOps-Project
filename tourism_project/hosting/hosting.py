from huggingface_hub import HfApi
import os

"""Script to upload the Streamlit app to Hugging Face Spaces."""

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_project/deployment",  # local folder containing app.py, requirements.txt, etc.
    repo_id="bhumitps/MLops",                 # Hugging Face Space name
    repo_type="space",                        # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

print("Deployment assets uploaded to Hugging Face Space: bhumitps/MLops")
