import random
import numpy as np

random.seed(42)

PATH = "data/sentences.txt"

test_data = []

with open(PATH, "r") as f:
    sentences = f.readlines()
    random.shuffle(sentences)

    test_data = sentences[:500]
    train_data = sentences[500:]
    with open("data/train.txt", "w") as f:
        f.writelines(train_data)

    test_data = np.array_split(test_data, 3)

    for idx, batch in enumerate(test_data):
        with open(f"data/{idx}.txt", "w") as f:
            f.writelines(batch)
