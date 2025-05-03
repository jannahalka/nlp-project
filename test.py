from transformers import AutoTokenizer, PreTrainedTokenizerFast

MODEL_NAME = "google-bert/bert-base-cased"

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True
)
