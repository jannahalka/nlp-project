import json
from torch.utils.data import Dataset


def convert_annotated_data_to_dict():
    data: dict[str, list[list[str]]] = {
        "tokens": [],
        "labels": [],
    }  # todo: convert long type def to a type

    with open("./data/annotated_data.jsonl") as f:
        for line in f.readlines():
            example = json.loads(line)
            labels = example["label"]

            example_tokens: list[str] = []
            example_labels: list[str] = []

            for label_info in labels:
                start: int = label_info[0]
                end: int = label_info[1]
                label: str = label_info[2]
                token: str = example["text"][start:end]
                example_tokens.append(token)
                example_labels.append(label)

            data["tokens"].append(example_tokens)
            data["labels"].append(example_labels)

    return data


gold_data = convert_annotated_data_to_dict()


class GoldDataset(Dataset):
    def __init__(self):
        self.data = convert_annotated_data_to_dict()

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        return self.data["tokens"][index], self.data["labels"][index]
