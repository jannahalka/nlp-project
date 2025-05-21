import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from .utils.hf_datasets import get_dataset
from .utils.labels import id2label, label2id
from .utils.tokenization import tokenize
from .utils.device import device
from .utils.model_evaluation import evaluate_model

dataset = get_dataset()
train, dev, test = dataset.values()

MODEL_NAME = "google-bert/bert-base-cased"
EPOCHS = 3
BATCH_SIZE = 64


config = AutoConfig.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), label2id=label2id, id2label=id2label
)
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True
)
model: torch.nn.Module = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, config=config
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}\n" + "-" * 40)
    model.train()

    train_loader = DataLoader(
        train,
        collate_fn=lambda batch: tokenize(batch, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    train_loader = tqdm(train_loader, total=len(train_loader))

    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            processed = batch_idx * BATCH_SIZE + batch["input_ids"].size(0)
            print(
                f"Epoch={epoch+1}; Loss: {loss.item():.4f} [{processed}/{len(train)}]"
            )

    dev_loader = DataLoader(
        dev,
        collate_fn=lambda batch: tokenize(batch, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    evaluate_model(model, dev_loader)
