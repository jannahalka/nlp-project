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

torch.manual_seed(42)


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

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    model_name, use_fast=True
)


def collate_fn(batch_sentences):
    enc = tokenizer(
        batch_sentences,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    enc["word_ids"] = enc.word_ids()

    return enc


dataloader = DataLoader(
    UnlabeledDataset("./data/unlabeled.txt"),
    collate_fn=collate_fn,
    batch_size=1,
    shuffle=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

import torch
import torch.nn.functional as F

threshold = 0.8  # only accept predictions with p ≥ 0.8

all_tokens = []
all_labels = []

for batch in dataloader:
    # alignment info
    word_ids = batch["word_ids"]

    # forward pass
    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    outputs = model(**inputs)
    logits = outputs.logits                            # (batch_size, seq_len, num_labels)
    probs  = F.softmax(logits, dim=-1)                 # convert to probabilities
    preds  = torch.argmax(logits, dim=-1)              # (batch_size, seq_len)

    # decode this sentence
    input_ids = inputs["input_ids"][0]                 # (seq_len,)
    tokens    = tokenizer.convert_ids_to_tokens(input_ids)

    sent_tokens = []
    sent_labels = []
    last_widx   = None

    for idx, (tok, lab, widx) in enumerate(zip(tokens, preds[0].cpu().tolist(), word_ids)):
        # skip special tokens
        if widx is None:
            continue

        # clean off BERT's '##' if it's a continuation
        clean = tok[2:] if tok.startswith("##") else tok

        # get this token's confidence
        conf = probs[0, idx, lab].item()

        if widx != last_widx:
            # new word: start fresh
            sent_tokens.append(clean)
            sent_labels.append(lab if conf >= threshold else -100)
            last_widx = widx
        else:
            # continuation: merge into the last token, keep its original label
            sent_tokens[-1] += clean

    all_tokens.append(sent_tokens)
    all_labels.append(sent_labels)
    break

goal = {"tokens": all_tokens, "labels": all_labels}
print(goal)

