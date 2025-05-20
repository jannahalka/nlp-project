from ..models.baseline_model.train import (
    BaselineModelTrainer,
    get_dataset,
    label2id,
    id2label,
)
from ..models.baseline_model.utils import device
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
import torch

torch.manual_seed(42)

dataset = get_dataset()

MODEL_NAME = "google-bert/bert-base-cased"

config = AutoConfig.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), label2id=label2id, id2label=id2label
)

trainer = BaselineModelTrainer(dataset)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)


def tokenize(batch):
    tokens = list(map(lambda b: b["tokens"], batch))
    labels = list(map(lambda b: b["labels"], batch))

    enc = tokenizer(
        tokens,
        max_length=128,
        padding="max_length",
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt",
    )

    # Align labels -> Mask subwords to -100
    aligned_labels = []

    for batch_index, original_labels in enumerate(labels):
        word_ids = enc.word_ids(batch_index=batch_index)

        IGNORE_LABEL = -100
        masked = []
        prev_word_id: int | None = None

        for word_id in word_ids:
            if word_id is None or word_id == prev_word_id:
                masked.append(IGNORE_LABEL)
            else:
                label = original_labels[word_id]
                masked.append(label)
            prev_word_id = word_id
        aligned_labels.append(masked)

    enc["labels"] = torch.tensor(aligned_labels)

    return enc


def test_train_loop():
    train, dev, test = dataset.values()

    dataloader = DataLoader(
        train,
        collate_fn=tokenize,
        batch_size=5,
        shuffle=True,
    )

    model.train()

    dataloader = tqdm(
        dataloader,
        total=len(dataloader),
    )

    batch = next(iter(dataloader))
    batch = {key: value.to(device) for key, value in batch.items()}
    output = model(**batch)
    loss = output.loss
    print(loss)
    loss.backward()
