from huggingface_hub import HfApi
import os
from .train import label2id

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/models/baseline",
    repo_id="jannahalka/nlp-project-baseline",
    repo_type="model",
)

