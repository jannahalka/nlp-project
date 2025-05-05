from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForTokenClassification,
)


# Returns (N, ) data
class UnlabeledDataset(Dataset):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path, delimiter="\t", header=None)
        self.df.columns = ["sentence"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> str:
        return self.df.iloc[index]["sentence"]


model_name = "jannahalka/nlp-project-baseline"

config = AutoConfig.from_pretrained(
    model_name,
    label2id={
        "O": "0",
        "B-PER": "1",
        "I-PER": "2",
        "B-LOC": "3",
        "I-LOC": "4",
        "B-ORG": "5",
        "I-ORG": "6",
    },
)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    model_name, use_fast=True
)


def collate_fn(batch):
    return tokenizer(
        batch,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )


dataloader = DataLoader(
    UnlabeledDataset("./data/unlabeled.txt"),
    collate_fn=collate_fn,
    batch_size=64,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

for batch in dataloader:
    batch = {key: value.to(device) for key, value in batch.items()}
    outputs = model(**batch)
    logits = (
        outputs.logits
    )  # Shape -> (64, 56, 7), where 64=batch_size, 56=max token lenght, 7=labels vector corresponding to each class's prediction
    # We want to look at the last dimensions, since that's where the predictions are
    preds = torch.argmax(logits, dim=-1)

    for i in range(batch["input_ids"].size(0)):
        for p in preds:
            print(p)
        break
