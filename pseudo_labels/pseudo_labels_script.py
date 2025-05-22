import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .models.baseline_model.datasets import UnlabeledDataset
from .models.baseline_model.utils import collate_sentences
from .utils.device import device
from .utils.model_evaluation import evaluate_model
from .utils.tokenization import tokenize
from .models.baseline_model.model import get_trained_model, get_trained_tokenizer
from .models.baseline_model.example_dataclass import Example
from .utils.gold_dataset import GoldDataset
from .utils.pseudo_train_dataset import PseudoTrainDataset


class PseudoLabels:
    def __init__(self):
        self.model: torch.nn.Module = get_trained_model()
        self.model.to(device)
        self.tokenizer = get_trained_tokenizer()
        self.confidence_threshold = 0.60

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def iteration(self, dataloader):
        self.model.eval()
        silver_data = {"tokens": [], "labels": []}
        special_ids = set(self.tokenizer.all_special_ids)

        for batch in tqdm(dataloader, desc="Pseudo-labeling"):
            batch_size = batch["input_ids"].size(0)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidences, pred_indices = torch.max(probs, dim=-1)

            for i in range(batch_size):
                token_ids = batch["input_ids"][i].cpu().tolist()
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                preds = pred_indices[i].cpu().tolist()
                confs = confidences[i].cpu().tolist()

                # filter out special tokens
                filtered_tokens, filtered_labels, filtered_confidences = [], [], []
                for tid, tok, idx, conf in zip(token_ids, tokens, preds, confs):
                    if tid in special_ids:
                        continue
                    filtered_tokens.append(tok)
                    filtered_labels.append(idx)
                    filtered_confidences.append(conf)

                ex = Example(filtered_tokens, filtered_labels, filtered_confidences)
                ex.merge_subwords()
                ex.mask(self.confidence_threshold)
                silver_data["tokens"].append(ex.tokens)
                silver_data["labels"].append(ex.labels)

        return silver_data

    def train(self, loader: DataLoader, lr: float, epochs=5):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        eval_loader = DataLoader(
            GoldDataset("./pseudo_labels/data/gold/annotated_dev.jsonl"),
            shuffle=False,
            batch_size=64,
            collate_fn=lambda batch: tokenize(batch, self.tokenizer),
        )

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Train Epoch {epoch}"):
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = self.model(**batch).loss
                if torch.isnan(loss):
                    print(f"Batch has all labels = -100? ", (batch['labels']==-100).all())
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            logging.info(f"→ Epoch {epoch} avg loss: {avg_loss:.4f}")

        evaluate_model(self.model, eval_loader)
        total = sum(len(labs) for labs in silver_data["labels"])
        kept = sum(sum(l != -100 for l in labs) for labs in silver_data["labels"])
        logging.info(f"  → token coverage: {kept/total:.1%}")


if __name__ == "__main__":
    ITERATIONS = 3

    pseudo = PseudoLabels()
    test_loader = DataLoader(
        GoldDataset("./pseudo_labels/data/gold/annotated_test.jsonl"),
        shuffle=False,
        batch_size=64,
        collate_fn=lambda batch: tokenize(batch, pseudo.tokenizer),
    )

    # ---- Pseudo‐label + training loop ----
    for i in range(ITERATIONS):
        logging.info(f"=== Iteration {i}/{--ITERATIONS} ===")

        unlabeled_loader = DataLoader(
            UnlabeledDataset("./pseudo_labels/data/sentences.txt"),
            batch_size=16,
            shuffle=True,
            collate_fn=collate_sentences,
        )
        silver_data = pseudo.iteration(unlabeled_loader)
        n_generated = len(silver_data["labels"])

        logging.info(f"Generated {n_generated} silver examples")

        train_loader = DataLoader(
            PseudoTrainDataset(silver_data),
            batch_size=64,
            shuffle=True,
            collate_fn=lambda batch: tokenize(batch, pseudo.tokenizer),
        )
        pseudo.train(train_loader, 2e-5)
        pseudo.confidence_threshold += 0.02

    evaluate_model(pseudo.model, test_loader)
