import pytest
from ..models.baseline_model.train import BaselineModelTrainer, get_dataset



def test_evalutation_loop():
    dataset = get_dataset()
    trainer = BaselineModelTrainer(dataset)
    trainer.test_loop()
