from torch.utils.data import Dataset
from .example_dataclass import Example
import pandas as pd


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
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {
            "inputs": self.examples[index].tokens,
            "labels": self.examples[index].labels,
        }
