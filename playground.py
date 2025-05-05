from datasets import load_dataset

dataset = load_dataset("jannahalka/nlp-project-data", trust_remote_code=True)

print(dataset['train'][0])
