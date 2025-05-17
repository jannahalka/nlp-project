from torch.utils.data import Dataset
from .example_dataclass import Example
from transformers import PreTrainedTokenizerFast
import pandas as pd
import torch


class UnlabeledDataset(Dataset):
    """
    Returns rows of sentences (not tokens).
    """

    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path, delimiter="\t", header=None)
        self.df.columns = ["sentence"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> str:
        return self.df.iloc[index]["sentence"]


class SilverDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
    ):
        self.encodings = tokenizer(
            [ex.tokens for ex in examples],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        self.labels = [ex.labels for ex in examples]
        self.examples = examples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item
