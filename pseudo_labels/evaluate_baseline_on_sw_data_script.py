from .utils.model_evaluation import evaluate_model
from .models.baseline_model.model import get_trained_model, get_trained_tokenizer
from .utils.gold_dataset import GoldDataset
from .utils.tokenization import tokenize
from .utils.device import device
from torch.utils.data import DataLoader

model = get_trained_model()
tokenizer = get_trained_tokenizer()

model.to(device)

gold_dataset = GoldDataset()
dataloader = DataLoader(
    gold_dataset,
    shuffle=False,
    batch_size=64,
    collate_fn=lambda batch: tokenize(batch, tokenizer),
)
if __name__ == "__main__":
    print("Running evaluation on the GoldDataset")
    evaluate_model(model, dataloader)
    print("Done")
