from torch.utils.data import DataLoader
from models.baseline_model.predict import predict, collate
from models.baseline_model.datasets import UnlabeledDataset

dataloader = DataLoader(
    UnlabeledDataset("./data/unlabeled.txt"),
    batch_size=2,
    collate_fn=collate,
    shuffle=True,
)

def test_predict():
    probs, preds = predict(dataloader)
    assert len(probs[0]) == len(preds[0]) # dimensions match
