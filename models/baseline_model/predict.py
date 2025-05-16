from datasets import DatasetDict, load_dataset, Dataset as DatasetHF
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForTokenClassification,
)
import torch.nn.functional as F
from .train import BaselineModelTrainer
from .datasets import UnlabeledDataset


def collate(batch_sentences):
    enc = baseline_tokenizer(
        batch_sentences,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    batch_word_ids = [enc.word_ids(i) for i in range(enc.input_ids.shape[0])]
    enc["word_ids"] = batch_word_ids
    return enc


torch.manual_seed(42)

iterations = 5
baseline_model_name = "jannahalka/nlp-project-baseline"
baseline_model = AutoModelForTokenClassification.from_pretrained(baseline_model_name)
baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name, use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(dataloader):
    baseline_model.to(device)
    baseline_model.eval()
    batch = next(iter(dataloader))  # todo: change to for loop
    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    outputs = baseline_model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)

    return probs, preds


def handle(tokens, preds, word_ids: list[int], probs):
    THRESHOLD = 0.90

    last_word_idx = None
    sent_tokens = []
    sent_labels = []

    for idx, (token, label, word_idx) in enumerate(
        zip(tokens, preds.squeeze().cpu().tolist(), word_ids)
    ):
        is_special_token = word_idx == None
        if is_special_token:
            continue
        # clean off BERT's '##' if it's a continuation
        clean = token[2:] if token.startswith("##") else token
        confidence = probs[0, idx, label].item()

        if word_idx != last_word_idx:
            # new word: start fresh
            sent_tokens.append(clean)
            sent_labels.append(label if confidence >= THRESHOLD else -100)
            last_word_idx = word_idx
        else:
            # continuation: merge into the last token, keep its original label
            sent_tokens[-1] += clean

    return sent_tokens, sent_labels


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

    def iterate(self):
        for i in range(self.iterations):
            dataset = self.predict()
            trainer = BaselineModelTrainer(dataset, model_name=self.model_name)
            trainer.train()
            self.model = trainer.get_model()
