from pathlib import Path

path = "data/baseline/en_ewt-ud-train.iob2"


def read_conll(path: str):
    """
    returns (N,2) matrix, where N = number of words in a conll file
    """
    data = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line:  # Comment
            continue

        data.append(line.split('\t')[1:3])

    return data

print(read_conll(path))

