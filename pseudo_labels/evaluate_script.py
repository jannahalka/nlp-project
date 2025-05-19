from utils.evaluate_model import evaluate_model
from utils.gold_dataset import GoldDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from models.baseline_model.model import get_trained_model
from models.baseline_model.utils import device
import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1).cpu()
    return accuracy_score(targets.cpu(), preds)


test_dataset = GoldDataset()
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model = get_trained_model().to(device)

results = evaluate_model(
    model,
    test_loader,
    device,
    criterion=torch.nn.CrossEntropyLoss(),
    metrics={'accuracy': accuracy}
)
print(f"Test loss: {results['loss']:.4f}, "
      f"Accuracy: {results['accuracy']*100:.2f}%")
