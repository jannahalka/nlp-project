from torch.utils.data import Dataset
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
