from datasets import Dataset, Features, Sequence, Value, ClassLabel, DatasetDict


def get_data(path):
    data = {"tokens": [], "ner_tags": []}
    labels = set()

    with open(path) as f:
        lines = f.readlines()

        # 1.index tokens, 2.index ner_tags
        temp = [[], []]
        for line in lines:
            line = line.strip()

            # Ignore comments
            if line.startswith("#"):
                continue

            if not line:
                data["tokens"].append(temp[0])
                data["ner_tags"].append(temp[1])
                temp = [[], []]
                continue

            parts = line.split()

            token = parts[1]
            ner = parts[2]
            labels.add(ner)

            temp[0].append(token)
            temp[1].append(ner)

    # dataset = Dataset.from_dict(data)
    # print(dataset.features['ner_tags'])

    labels = sorted(labels)

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=list(labels))),
        }
    )

    labels = {i: idx for (idx, i) in enumerate(labels)}

    for sentence in data["ner_tags"]:
        for idx, word in enumerate(sentence):
            sentence[idx] = labels[word]

    return Dataset.from_dict(data, features=features)

train = get_data("data/baseline/en_ewt-ud-train.iob2")
dev = get_data("data/baseline/en_ewt-ud-dev.iob2")
test = get_data("data/baseline/en_ewt-ud-test-masked.iob2")

dataset = DatasetDict({
    "train": train,
    "validation": dev,
    "test": test
})
