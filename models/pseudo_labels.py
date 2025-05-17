import torch
from torch.utils.data import DataLoader
from .baseline_model.datasets import SilverDataset, UnlabeledDataset
from .baseline_model.utils import collate_sentences, device
from .baseline_model.model import get_trained_model, get_trained_tokenizer
from .baseline_model.example_dataclass import Example

torch.manual_seed(42)

BATCH_SIZE = 1  # todo: Change to 64 in prod

dataloader = DataLoader(
    UnlabeledDataset("./data/sentences.txt"),
    collate_fn=collate_sentences,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = get_trained_model()
tokenizer = get_trained_tokenizer()
special_ids = set(tokenizer.all_special_ids)

silver_data: list[Example] = []

batch = next(iter(dataloader))  # change to loop in prod
batch = {key: value.to(device) for key, value in batch.items()}

outputs = model(**batch)
probs = torch.softmax(outputs.logits, dim=-1)
confidences, pred_indices = torch.max(probs, dim=-1)

token_ids = batch["input_ids"][0].cpu().tolist()  # [seq_len]
tokens = tokenizer.convert_ids_to_tokens(token_ids)  # [seq_len]
pred_indices = pred_indices[0].cpu().tolist()  # [seq_len]
confidence_scores = confidences[0].cpu().tolist()


filtered_tokens = []
filtered_labels = []
filtered_confidences = []

for tok_id, tok, idx, conf in zip(token_ids, tokens, pred_indices, confidence_scores):
    if tok_id in special_ids:
        continue

    filtered_tokens.append(tok)
    filtered_labels.append(idx)
    filtered_confidences.append(conf)

example = Example(filtered_tokens, filtered_labels, filtered_confidences)
example.merge_subwords()
example.mask()
silver_data.append(example)

dataset = SilverDataset(silver_data)
silver_dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
