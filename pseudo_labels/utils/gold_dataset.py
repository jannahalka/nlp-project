import json
from torch.utils.data import Dataset
from ..models.baseline_model.model import get_trained_tokenizer
from .labels import label2id


def convert_annotated_data_to_dict():
    data = {
        "tokens": [],
        "labels": [],
    }  # todo: convert long type definition to a type of its own

    with open("./pseudo_labels/data/annotated_data.jsonl") as f:
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
    def __init__(self):
        self.data = convert_annotated_data_to_dict()
        self.tokenizer = get_trained_tokenizer()

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        tokens = self.data["tokens"][index]
        enc = self.tokenizer(
            tokens,
            max_length=128,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )

        return enc, self.data["labels"][index]
