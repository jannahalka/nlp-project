from .model import get_trained_tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_sentences(batch_sentences):
    """
    Used to tokenize raw sentences
    """
    enc = get_trained_tokenizer()(
        batch_sentences,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    batch_word_ids = [enc.word_ids(i) for i in range(enc.input_ids.shape[0])]
    enc["word_ids"] = batch_word_ids
    return enc
