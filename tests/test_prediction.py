from torch.utils.data import DataLoader
from models.baseline_model.predict import predict
from models.baseline_model.datasets import UnlabeledDataset
from models.baseline_model.utils import collate_sentences

TEST_BATCH_SIZE = 2

dataloader = DataLoader(
    UnlabeledDataset("./data/unlabeled.txt"),
    batch_size=TEST_BATCH_SIZE,
    collate_fn=collate_sentences,
)

def test_predict():
    probs, preds = predict(dataloader)
    print(probs)
    assert len(probs[0]) == len(preds[0]) # dimensions match
