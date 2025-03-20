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
processed_test = dataset["test"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=["ner_tags"],
    desc="Tokenizing test dataset",
)

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
        batch = {k: v.to(device) for k, v in batch.items()}
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

# Assuming you have a `test_dataset` similar to your train and dev datasets.
# For example, you might have:
# test_dataset = processed_raw_datasets["test"]

def predict_and_align(example, model, tokenizer, label_list, device):
    # Tokenize the example (a dict with "tokens") and get tensor inputs.
    tokenized_example = tokenizer(
        example["tokens"],
        max_length=128,
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt"
    )

    # Get the word_ids to map predictions back to tokens.
    # Note: word_ids is a list with the same length as the tokenized inputs.
    word_ids = tokenized_example.word_ids(batch_index=0)

    # Move inputs to device
    tokenized_example = {k: v.to(device) for k, v in tokenized_example.items()}

    # Get model predictions (logits) and convert to label indices
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_example)
    logits = outputs.logits  # shape: [1, seq_len, num_labels]
    predictions = torch.argmax(logits, dim=-1)[0].tolist()  # for one example

    # Align predictions: Only keep the prediction for the first subword of each token.
    aligned_predictions = []
    previous_word_idx = None
    for word_idx, pred in zip(word_ids, predictions):
        if word_idx is None:
            continue  # skip special tokens
        if word_idx != previous_word_idx:
            aligned_predictions.append(label_list[pred])
            previous_word_idx = word_idx
    return aligned_predictions

# Now, iterate over all test examples to get predictions.
test_predictions = []
for example in processed_test:
    pred_labels = predict_and_align(example, model, tokenizer, label_list, device)
    test_predictions.append({
        "tokens": example["tokens"],
        "predicted_labels": pred_labels
    })

# Write predictions to an output file in IOB2 format.
# Each line will have the token and its predicted label separated by a space,
# and sentences are separated by a blank line.
with open("predictions.iob2", "w") as f:
    for item in test_predictions:
        tokens = item["tokens"]
        labels = item["predicted_labels"]
        for token, label in zip(tokens, labels):
            f.write(f"{token}\t{label}\n")
        f.write("\n")

