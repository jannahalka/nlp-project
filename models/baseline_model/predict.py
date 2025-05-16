import torch
from torch.utils.data import DataLoader
from .utils import device
from .model import get_trained_model
from typing import List
from dataclasses import dataclass


@dataclass
class Example:
    THRESHOLD = 0.90
    IGNORE_LABEL = -100

    tokens: List[str]
    labels: List[int]
    scores: List[float]

    def mask(self) -> None:
        """
        Replace any label whose corresponding score is below THRESHOLD with -100.
        """
        if len(self.labels) != len(self.scores):
            raise ValueError("labels and scores must be the same length")

        for idx, score in enumerate(self.scores):
            if score < self.THRESHOLD:
                self.labels[idx] = self.IGNORE_LABEL


def get_predictions():
    """
    Predicts the outputs of dataloader
    Returns: Tuple[Tensor, Tensor], where Tuple[0]=probabilites, and Tuple[1]=predictions
    """
    model = get_trained_model()
    model.to(device)
    model.eval()
    batch = next(iter(dataloader))  # todo: change to for loop
    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)

    return preds


def get_probabilities(logits):
    return torch.softmax(logits, dim=-1)


def create_silver_dataset() -> List[Example]:
    return [
        Example(
            tokens=["Yoda", "is", "a", "Jedi", "."],
            labels=[1, 0, 0, 3, 0],
            scores=[0.88, 0.92, 0.99, 0.90, 0.99],
        ),
        Example(
            tokens=["Darth", "Vader", "is", "a" "Sith", "."],
            labels=[1, 2, 0, 0, 3, 0],
            scores=[0.88, 0.92, 0.99, 0.90, 0.99],
        ),
    ]


# todo: create_silver_dataset() will return data in the format below
# Raw sentences: ["Yoda is a Jedi.", "Darth Vader is a Sith."]
# Output from create_silver_dataset(): [[["Yoda", "is", "a", "Jedi", "."], [1, 0, 0, 3, 0], [0.88, 0.92, 0.99, 0.90, 0.99]], ["Darth", "Vader", "is", "a", "Sith", "."]]
