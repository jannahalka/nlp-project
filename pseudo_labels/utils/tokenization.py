import torch
from transformers import PreTrainedTokenizerFast


def tokenize(batch, tokenizer: PreTrainedTokenizerFast):
    tokens = [b["tokens"] for b in batch]
    labels = [b["labels"] for b in batch]

    enc = tokenizer(
        tokens,
        max_length=128,
        padding="max_length",
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt",
    )

    aligned_labels = []

    for batch_index, original_labels in enumerate(labels):
        word_ids = enc.word_ids(batch_index=batch_index)
        IGNORE_LABEL = -100
        masked = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None or word_id == prev_word_id:
                masked.append(IGNORE_LABEL)
            else:
                masked.append(original_labels[word_id])
            prev_word_id = word_id
        aligned_labels.append(masked)

    enc["labels"] = torch.tensor(aligned_labels)

    return enc
