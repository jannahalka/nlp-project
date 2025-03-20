from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from data import dataset


MODEL_NAME = "google-bert/bert-base-cased"
LR = 2e-5
EPOCHS = 3
LABEL_COLUMN_NAME = "ner_tags"

label_list = dataset["train"].features[LABEL_COLUMN_NAME].feature.names


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(label_list))


def tokenize_and_align_labels(examples):
    """
    For each example, tokenize the list of tokens and align the original labels
    to the resulting subwords. Tokens can be split into multiple subwords, so we mark
    the "extra" subwords with -100 to ignore them in the loss.
    """

    # 1) Tokenize
    # 'is_split_into_words=True' tells the tokenizer each item in the list is already a separate word/token.
    tokenized_inputs = tokenizer(
        examples["tokens"],  # e.g., examples["tokens"]
        max_length=128,
        padding=False,
        truncation=True,
        is_split_into_words=True,
    )

    # 2) Prepare a new "labels" list aligned to the subword tokens
    all_labels = []

    # examples[label_column_name] might look like: [0, 0, 1, 2, ...] for each token
    for batch_index, labels in enumerate(examples[LABEL_COLUMN_NAME]):
        # 'word_ids()' returns a list the same length as the subword-tokens,
        # each entry telling you which 'word' or token it came from
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)

        label_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # e.g. special tokens or padding
                label_ids.append(-100)
            elif word_id == prev_word_id:
                # subword token of the same word => ignore
                label_ids.append(-100)
            else:
                # new subword, so use the label for the original token
                label_ids.append(labels[word_id])

            prev_word_id = word_id

        all_labels.append(label_ids)

    # 3) Attach the new "labels" to our tokenized inputs
    tokenized_inputs["labels"] = all_labels

    # 4) Return the updated dictionary
    return tokenized_inputs


processed_raw_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_raw_datasets["train"]
eval_dataset = processed_raw_datasets["validation"]

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)

data_collator = DataCollatorForTokenClassification(tokenizer)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=8
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=8)

# Move model to device (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create optimizer (e.g. AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

running_loss = 0.0
last_loss = 0.0

for epoch in range(EPOCHS):
    model.train(True)
    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Training Epoch {epoch+1}",
    )

    for step, batch in pbar:  # Iterate over batches of data
        optimizer.zero_grad()  # Clear gradients from the previous iteration

        outputs = model(**batch)  # Forward pass through the model
        loss = outputs.loss
        loss.backward()  # Compute gradients (backpropagation)
        optimizer.step()  # Update model parameters

        running_loss += loss.item()

        if step % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(step + 1, last_loss))
            running_loss = 0.0
