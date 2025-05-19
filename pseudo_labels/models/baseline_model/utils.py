from .model import get_trained_tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_sentences(batch_sentences):
    """
    Used to tokenize raw sentences
    """

    return get_trained_tokenizer()(
        batch_sentences,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
