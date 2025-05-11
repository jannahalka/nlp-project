from datasets import DatasetDict, load_dataset, Dataset as DatasetHF
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForTokenClassification,
)
import torch.nn.functional as F
from .train import BaselineModelTrainer

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


# goal = {"tokens": all_tokens, "labels": all_labels}


class PseudoLabels:
    def __init__(self, iterations=5, model_name="jannahalka/nlp-project-baseline"):
        self.iterations = iterations
        self.model_name = model_name
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = DataLoader(
            UnlabeledDataset("./data/unlabeled.txt"),
            collate_fn=self.collate_fn,
            batch_size=64,
            shuffle=True,
        )

    def collate_fn(self, batch_sentences):
        is_split = isinstance(batch_sentences[0], (list, tuple))

        enc = self.tokenizer(
            batch_sentences,
            padding="longest",
            truncation=True,
            max_length=128,
            is_split_into_words=is_split,
            return_tensors="pt",
        )
        # for fast tokenizers, word_ids() must be called per sample:
        batch_word_ids = [enc.word_ids(i) for i in range(enc.input_ids.shape[0])]
        enc["word_ids"] = batch_word_ids
        return enc

    def predict(self) -> DatasetDict:
        self.model.to(self.device)
        self.model.eval()
        threshold = 0.8  # only accept predictions with p ≥ 0.8

        all_tokens = []
        all_labels = []

        for batch in self.dataloader:
            # alignment info
            word_ids = batch["word_ids"]

            # forward pass
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = self.model(**inputs)
            logits = outputs.logits  # (batch_size, seq_len, num_labels)
            probs = F.softmax(logits, dim=-1)  # convert to probabilities
            preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

            # decode this sentence
            input_ids = inputs["input_ids"][0]  # (seq_len,)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            sent_tokens = []
            sent_labels = []
            last_widx = None

            for idx, (tok, lab, widx) in enumerate(
                zip(tokens, preds[0].cpu().tolist(), word_ids)
            ):
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

        dataset = DatasetDict(
            {
                "train": DatasetHF.from_dict(
                    {"tokens": all_tokens, "labels": all_labels}
                ),
                "test": DatasetHF.from_dict(
                    {"tokens": all_tokens, "labels": all_labels}
                ),
                "dev": DatasetHF.from_dict(
                    {"tokens": all_tokens, "labels": all_labels}
                ),
            }
        )
        train_dataset = dataset["train"]

        self.dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=self.collate_fn,
            batch_size=64,
            shuffle=True,
        )

        return dataset

    def iterate(self):
        for i in range(self.iterations):
            dataset = self.predict()
            trainer = BaselineModelTrainer(dataset, model_name=self.model_name)
            trainer.train()
            self.model = trainer.get_model()


pl = PseudoLabels()
pl.iterate()
