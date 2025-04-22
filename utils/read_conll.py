from pathlib import Path
import csv
from datasets import load_dataset, Dataset

path = "data/baseline/en_ewt-ud-train.iob2"


def read_conll(path: str):
    """
    returns (N,2) matrix, where N = number of words in a conll file
    """
    data = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line:  # Ignore bs
            continue

        data.append(line.split("\t")[1:3])

    return data


def write_conll_to_csv(io2_file_path: str, output_path: str):
    """
    Util function for translating io2 files to csv
    """

    data = read_conll(io2_file_path)
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = ["token", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{"token": i[0], "label": i[1]} for i in data])


paths = [
    {"path": "data/baseline/en_ewt-ud-train.iob2", "label": "train"},
    {"path": "data/baseline/en_ewt-ud-dev.iob2", "label": "dev"},
    {"path": "data/baseline/en_ewt-ud-test-masked.iob2", "label": "test"},
]

for obj in paths:
    write_conll_to_csv(obj["path"], f"data/baseline/{obj["label"]}.csv")

data = {
    "train": "data/baseline/train.csv",
    "dev": "data/baseline/dev.csv",
    "test": "data/baseline/test.csv",
}

dataset = load_dataset("csv", data_files=data)
