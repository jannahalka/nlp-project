from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
}


def get_dataset():
    dataset = load_dataset("jannahalka/nlp-project-data", trust_remote_code=True)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict, got {type(dataset).__name__!r}")

    return dataset


class BaselineModelTrainer:
    def __init__(
        self,
        dataset: DatasetDict,
        model_name="google-bert/bert-base-cased",
        learning_rate=2e-5,
        batch_size=8,
        epochs=3,
    ):
        self.batch_size = batch_size
        self.epochs = epochs

        train, dev, test = dataset.values()
        self.train_dataset, self.dev_dataset, self.test_dataset = train, dev, test

        labels = label2id.values()
        config = AutoConfig.from_pretrained(model_name, num_labels=len(labels))

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(self):
        self.prepare_data()

        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            # TODO: self.test_epoch()

    def train_epoch(self, epoch: int):
        self.model.train()

        train_loader = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Training Epoch {epoch + 1}",
        )

        for batch in train_loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}

            output = self.model(**batch)
            loss = output.loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def prepare_data(self):
        self.train_dataset = self.train_dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Running tokenizer on the dataset",
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
        )

    def tokenize(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            max_length=128,  # Max length of a batch
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )
        labels = self.align_labels(examples, tokenized_inputs)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def align_labels(self, examples, tokenized_inputs):
        labels = []

        for batch_index, original_labels in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            masked_labels = self.create_labels_mask(word_ids, original_labels)
            labels.append(masked_labels)

        return labels

    def create_labels_mask(self, word_ids, original_labels):
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
        return masked

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


dataset = get_dataset()

trainer = BaselineModelTrainer(dataset)
trainer.train()
trainer.save("./models/baseline_model/output")
