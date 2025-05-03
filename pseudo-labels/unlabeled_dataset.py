from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "google-bert/bert-base-cased"


class UnlabeledDataset(Dataset):
    """
    Used to create silver data from our unlabeled dataset
    """

    def __init__(self, dataset_path, transform=None):
        self.df = pd.read_csv(dataset_path, delimiter="\t", header=None)
        self.df.columns = ["sentence"]
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.iloc[index].sentence

        enc = self.tokenizer(
            sentence,
            max_length=128,
            padding=False,
            truncation=True,
        )
        print(enc.word_ids)

        return self.df.iloc[index]


d = UnlabeledDataset("./data/unlabeled.txt")
d[0]
