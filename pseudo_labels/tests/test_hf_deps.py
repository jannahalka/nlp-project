# Exploration of Huggingface
from ..models.baseline_model.model import get_trained_model

model = get_trained_model()
label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}


def test_label2id():
    assert label2id == model.config.label2id
