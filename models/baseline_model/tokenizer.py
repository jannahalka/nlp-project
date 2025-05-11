
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForTokenClassification,
)

model_name = "jannahalka/nlp-project-baseline"

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    model_name, use_fast=True
)
