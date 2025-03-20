from transformers import AutoModelForTokenClassification, AutoConfig
import torch
from data import dataset

MODEL_NAME = "google-bert/bert-base-cased"
LABEL_COLUMN_NAME = "ner_tags"

label_list = dataset["train"].features[LABEL_COLUMN_NAME].feature.names

config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(label_list))
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)
# Load the trained weights
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()  # Set the model to evaluation mode

