from torch.utils.data import Dataset
import pandas as pd


class UnlabeledDataset(Dataset):
    """
    Used to create silver data from our unlabeled dataset
    """

    def __init__(self, dataset_path, transform=None):
        self.df = pd.read_csv(dataset_path, delimiter="\t", header=None)
        self.df.columns = ["sentence"]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        print(index)
        return self.df.iloc[index]
