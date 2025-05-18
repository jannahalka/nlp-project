import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from .baseline_model.datasets import SilverDataset, UnlabeledDataset
from .baseline_model.utils import collate_sentences, device
from .baseline_model.model import get_trained_model, get_trained_tokenizer
from .baseline_model.example_dataclass import Example


def pseudo_label_iteration(model, tokenizer, unlabeled_loader):
    """
    Run the model in eval mode over the unlabeled data,
    produce and return a list of silver‐label Examples.
    """
    model.eval()
    silver_data = []
    special_ids = set(tokenizer.all_special_ids)

    for batch in tqdm(unlabeled_loader, desc="Pseudo-labeling"):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward + softmax
        with torch.no_grad():
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, pred_indices = torch.max(probs, dim=-1)

        # Only take the first sequence in the batch (we use batch_size=1 for pseudo‐labeling)
        token_ids = batch["input_ids"][0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        preds = pred_indices[0].cpu().tolist()
        confs = confidences[0].cpu().tolist()

        # Filter out special tokens, gather subword‐merged, masked Example
        filtered_tokens, filtered_labels, filtered_confidences = [], [], []
        for tok_id, tok, idx, conf in zip(token_ids, tokens, preds, confs):
            if tok_id in special_ids:
                continue
            filtered_tokens.append(tok)
            filtered_labels.append(idx)
            filtered_confidences.append(conf)

        ex = Example(filtered_tokens, filtered_labels, filtered_confidences)
        ex.merge_subwords()
        ex.mask()
        silver_data.append(ex)

    return silver_data


def train_on_silver(model, tokenizer, silver_data, batch_size, lr, epochs):
    """
    Train the model on the provided silver_data for a given number of epochs.
    Logs loss per batch and average loss per epoch.
    """
    model.train()
    collator = DataCollatorForTokenClassification(tokenizer)
    dataset = SilverDataset(silver_data, tokenizer, max_length=128)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss = loss.item()
            total_loss += batch_loss
            logging.info(
                f"[Epoch {epoch} | Step {step+1}/{len(loader)}]  Loss: {batch_loss:.4f}"
            )

        avg_loss = total_loss / len(loader)
        logging.info(f"→ Epoch {epoch} completed. Average Loss: {avg_loss:.4f}\n")


if __name__ == "__main__":
    # ---- Config ----
    torch.manual_seed(42)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Hyper-parameters
    LR = 2e-5
    PSEUDO_ITERS = 3  # how many times to re-label + train
    TRAIN_EPOCHS = 2
    BATCH_SIZE = 64

    # ---- Prepare data & model ----
    tokenizer = get_trained_tokenizer()
    model = get_trained_model().to(device)

    unlabeled_ds = UnlabeledDataset("./data/sentences.txt")
    unlabeled_ld = DataLoader(
        unlabeled_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sentences,
    )

    # ---- Pseudo-label loop ----
    for iteration in range(1, PSEUDO_ITERS + 1):
        logging.info(f"=== Pseudo-Label Iteration {iteration}/{PSEUDO_ITERS} ===")
        silver_examples = pseudo_label_iteration(model, tokenizer, unlabeled_ld)
        logging.info(f"Generated {len(silver_examples)} silver examples.")

        train_on_silver(
            model,
            tokenizer,
            silver_examples,
            batch_size=BATCH_SIZE,
            lr=LR,
            epochs=TRAIN_EPOCHS,
        )

    # ---- Save final model ----
    save_dir = "./models/pseudo_trained"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logging.info(f"Training complete. Model saved to {save_dir}")
