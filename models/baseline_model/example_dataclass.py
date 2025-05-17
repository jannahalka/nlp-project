import torch
from torch.utils.data import Dataset
from ..errors import InvalidDimensionException
from typing import List
from dataclasses import dataclass


threshold = 0.90


@dataclass
class Example:
    IGNORE_LABEL = -100

    tokens: List[str]
    labels: List[int]
    confidences: List[float]

    def __post_init__(self):
        if not self.tokens or not self.labels or not self.confidences:
            raise ValueError("tokens, labels and confidences must not be empty")
        n = len(self.tokens)
        if len(self.labels) != n or len(self.confidences) != n:
            raise InvalidDimensionException(
                "Dimension of either `confidences` or `labels` don't match with `tokens`"
            )

    def mask(self) -> None:
        """
        Replace any label whose corresponding score is below THRESHOLD with -100.
        """
        if len(self.labels) != len(self.confidences):
            raise ValueError("labels and scores must be the same length")

        for idx, confidence in enumerate(self.confidences):
            if confidence < threshold:
                self.labels[idx] = self.IGNORE_LABEL

    def merge_subwords(self) -> None:
        """
        Merge tokens starting with '##' into the previous token, dropping their labels and confidences.
        e.g. ['Yo', '##da', 'is'] -> ['Yoda', 'is']
        """
        new_tokens: List[str] = []
        new_labels: List[int] = []
        new_confidences: List[float] = []

        for tok, lab, conf in zip(self.tokens, self.labels, self.confidences):
            if tok.startswith("##"):
                clean = tok[2:]
                if not new_tokens:
                    raise ValueError("Cannot merge a subword at the beginning of the sequence")
                new_tokens[-1] += clean
            else:
                new_tokens.append(tok)
                new_labels.append(lab)
                new_confidences.append(conf)

        self.tokens = new_tokens
        self.labels = new_labels
        self.confidences = new_confidences

    def prepare_for_training(self) -> Dataset:
        self.merge_subwords()
        self.mask()

