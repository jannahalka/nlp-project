from datasets import load_dataset, DatasetDict


def get_dataset():
    dataset = load_dataset("jannahalka/nlp-project-data", trust_remote_code=True)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict, got {type(dataset).__name__!r}")

    return dataset
