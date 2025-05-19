from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForTokenClassification,
    BertForTokenClassification,
)  # todo: figure out why is there an import err msg

MODEL_NAME = "jannahalka/nlp-project-baseline"


def get_trained_model():
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    if not isinstance(model, BertForTokenClassification):
        raise Exception()  # todo: if time custom exception

    return model


def get_trained_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise Exception()  # todo: if time custom exception

    return tokenizer
