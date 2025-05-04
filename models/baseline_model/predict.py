import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForTokenClassification,
)


class UnlabeledDataset(Dataset):
    """
    Used to create silver data from our unlabeled dataset
    """

    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path, delimiter="\t", header=None)
        self.df.columns = ["sentence"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]["sentence"]


model_name = "jannahalka/nlp-project-baseline"

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


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
    batch_size=32,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
pseudo_inputs, pseudo_labels = [], []


with torch.no_grad():
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)

        keep_mask = max_probs >= 0.90

        for i in range(batch["input_ids"].size(0)):
            if keep_mask[i].all():  # or some percentage of tokens in the sentence
                pseudo_inputs.append(batch["input_ids"][i])
                pseudo_labels.append(preds[i])
