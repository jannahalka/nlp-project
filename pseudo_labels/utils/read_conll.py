from pathlib import Path
from datasets import Features, Sequence, Value
import json

# TODO: Refactor this shitshow (whole file)
path = "data/baseline/en_ewt-ud-train.iob2"

label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
}

id2label = {value: key for key, value in label2id.items()}


labels = set()


def read_conll(path: str):
    data = {"tokens": [], "labels": []}
    temp = [[], []]

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue

        if not line:
            data["tokens"].append(temp[0])
            data["labels"].append(temp[1])
            temp = [[], []]
            continue

        valid = line.split("\t")[1:3]
        temp[0].append(valid[0])
        temp[1].append(label2id[valid[1]])
        labels.add(label2id[valid[1]])

    return data


def convert_io2_to_json(io2_path, output_path):
    data = read_conll(io2_path)
    data_json = json.dumps(data, indent=2)

    with open(output_path, "w") as f:
        f.write(data_json)


convert_io2_to_json("data/baseline/en_ewt-ud-train.iob2", "data/baseline/train.json")
convert_io2_to_json("data/baseline/en_ewt-ud-dev.iob2", "data/baseline/dev.json")
convert_io2_to_json(
    "data/baseline/en_ewt-ud-test-masked.iob2", "data/baseline/test.json"
)

data_files = {
    "train": "data/baseline/train.json",
    "dev": "data/baseline/dev.json",
    "test": "data/baseline/test.json",
}

features = Features(
    {
        "tokens": Sequence(feature=Value("string")),
        "labels": Sequence(feature=Value("int64")),
    }
)
