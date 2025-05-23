import json
from torch.utils.data import Dataset
from ..models.baseline_model.model import get_trained_tokenizer
from .labels import label2id


def convert_annotated_data_to_dict(ds_path: str):
    data = {
        "tokens": [],
        "labels": [],
    }  # todo: convert long type definition to a type of its own
    with open(ds_path) as f:
        for line in f.readlines():
            example = json.loads(line)
            labels = example["label"]

            example_tokens: list[str] = []
            example_labels: list[int] = []

            for label_info in labels:
                start: int = label_info[0]
                end: int = label_info[1]
                label: str = label_info[2]
                token: str = example["text"][start:end]
                example_tokens.append(token)
                example_labels.append(label2id[label])

            data["tokens"].append(example_tokens)
            data["labels"].append(example_labels)

    return data


class GoldDataset(Dataset):
    def __init__(self, ds_path: str):
        self.data = convert_annotated_data_to_dict(ds_path)
        self.tokenizer = get_trained_tokenizer()

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        tokens = self.data["tokens"][index]

        return {"tokens": tokens, "labels": self.data["labels"][index]}
