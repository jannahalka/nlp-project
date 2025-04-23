from pathlib import Path
import json

from datasets import load_dataset, Features, Sequence, ClassLabel, Value

path = "data/baseline/en_ewt-ud-train.iob2"

labels = set()


def read_conll(path: str):
    """
    returns (N,2) matrix, where N = number of words in a conll file
    """
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
        temp[1].append(valid[1])
        labels.add(valid[1])

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
        "labels": Sequence(feature=ClassLabel(names=list(labels))),
    }
)

dataset = load_dataset("json", data_files=data_files, features=features)
dataset.push_to_hub("jannahalka/nlp-project-data", private=True)
