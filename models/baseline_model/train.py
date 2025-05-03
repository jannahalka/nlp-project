from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

MODEL = "google-bert/bert-base-cased"
LR = 2e-5
EPOCHS = 3
LABEL_COLUMN = "labels"
BATCH_SIZE = 8
IGNORE_LABEL = -100


def get_dataset():
    dataset = load_dataset("jannahalka/nlp-project-data", trust_remote_code=True)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict, got {type(dataset).__name__!r}")

    return dataset


dataset = get_dataset()
train, dev, test = dataset.values()

labels: list[str] = train.features["labels"].feature.names

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
config = AutoConfig.from_pretrained(MODEL, num_labels=len(labels))
model = AutoModelForTokenClassification.from_pretrained(MODEL, config=config)


def create_labels_mask(word_ids: list[int | None], original_labels):
    masked = []
    prev_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None or word_id == prev_word_id:
            masked.append(IGNORE_LABEL)
        else:
            label = original_labels[word_id]  # Original label of a subword
            masked.append(label)  # New subword, use original label

        prev_word_id = word_id
    return masked


def align_batch_labels(examples, tokenized_inputs: BatchEncoding):
    masked_labels = []

    for batch_index, original_labels in enumerate(examples[LABEL_COLUMN]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        masked_labels = create_labels_mask(word_ids, original_labels)
        masked_labels.append(masked_labels)

    return masked_labels


def tokenize(examples):  # TODO: Fix the type of `examples`
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=128,  # Max length of a batch
        padding=False,
        truncation=True,
        is_split_into_words=True,
    )
    labels = align_batch_labels(examples, tokenized_inputs)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=train.column_names,  # Remove un-tokenized data
    desc="Running tokenizer on the dataset",
)

train, dev, test = dataset.values()

data_collator = DataCollatorForTokenClassification(tokenizer)

train_dataloader = DataLoader(
    train, shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def show_loss(loss, step, batch, size):
    if step % 100 == 0:
        loss, current = loss.item(), step * BATCH_SIZE + len(batch)
        print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_loop(model, epoch, train_loader):
    size = len(train_loader.dataset)

    model.train()

    train_loader = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Training Epoch {epoch + 1}",
    )

    for step, batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(**batch)
        loss = output.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        show_loss(loss, step, batch, size)


for epoch in range(EPOCHS):
    train_loop(model, train_dataloader, epoch)


# TODO: Figure out `model` type
def save_model(model, tokenizer: PreTrainedTokenizerFast, output_path: str):
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


save_model(model, tokenizer, "./models/baseline_model/output")
