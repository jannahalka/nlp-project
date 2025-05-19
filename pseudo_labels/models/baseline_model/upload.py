from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./pseudo_labels/models/output",
    repo_id="jannahalka/nlp-project-baseline",
    repo_type="model",
)

