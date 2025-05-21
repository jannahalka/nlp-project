import torch
from torch.utils.data import DataLoader
from .device import device
from sklearn.metrics import classification_report
from .labels import id2label

def map_id_to_label(batch_ids):
    return [id2label[i] for i in batch_ids]

def evaluate_model(model: torch.nn.Module, eval_dataloader: DataLoader):
    model.eval()
    total_loss = 0.0
    all_label_ids = []
    all_pred_ids = []

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu()
            labels = batch["labels"].cpu()

            # align subword predictions back to word-level
            for i in range(labels.size(0)):
                word_ids = batch.word_ids(i)
                prev_word_id = None
                for token_idx, word_id in enumerate(word_ids):
                    if word_id is not None and word_id != prev_word_id:
                        all_label_ids.append(labels[i, token_idx].item())
                        all_pred_ids.append(preds[i, token_idx].item())
                    prev_word_id = word_id

    filtered = [(l, p) for l, p in zip(all_label_ids, all_pred_ids) if l != -100]
    true_ids, pred_ids = zip(*filtered)

    y_true = map_id_to_label(true_ids)
    y_pred = map_id_to_label(pred_ids)

    avg_loss = total_loss / len(eval_dataloader)
    print(f"Validation loss: {avg_loss:.4f}")
    print(classification_report(y_true, y_pred))

