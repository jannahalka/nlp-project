from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from torch.utils.data import DataLoader

model_name = "google-bert/bert-base-cased"

# dataset handling
label_list = []  # TODO

config = AutoConfig.from_pretrained(model_name, num_labels=len(label_list))

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

data_collator = DataCollatorForTokenClassification(tokenizer)

train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=8)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=8)
