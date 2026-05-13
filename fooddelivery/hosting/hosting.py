import os
from huggingface_hub import HfApi

# Read tokens from environment
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_KEY = os.getenv("GROQ_KEY")

# Validate env vars
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing")

if not GROQ_KEY:
    raise ValueError("GROQ_KEY is missing")

# Authenticate
api = HfApi(token=HF_TOKEN)

repo_id = "yashaswinigopalan/food-delivery-chatbot"

# Upload files to Hugging Face Space
api.upload_folder(
    folder_path="fooddelivery/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo=""
)

# Add secret to Hugging Face Space
api.add_space_secret(
    repo_id=repo_id,
    key="GROQ_KEY",
    value=GROQ_KEY
)

print("Deployment successful.")
