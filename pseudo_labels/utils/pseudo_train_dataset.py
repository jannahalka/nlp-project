from torch.utils.data import Dataset
from .gold_dataset import convert_annotated_data_to_dict


class PseudoTrainDataset(Dataset):
    def __init__(self, silver_data: dict):
        gold_data = convert_annotated_data_to_dict(
            "./pseudo_labels/data/gold/annotated_train.jsonl"
        )
        all_tokens = silver_data["tokens"] + gold_data["tokens"]
        all_labels = silver_data["labels"] + gold_data["labels"]
        self.data = {"tokens": all_tokens, "labels": all_labels}

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        return {
            "tokens": self.data["tokens"][index],
            "labels": self.data["labels"][index],
        }
