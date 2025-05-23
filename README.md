# Domain-Specific Named Entity Recognition for the Star Wars Universe

## Project Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Running Scripts
1. Baseline Model Training
```bash
python3 -m pseudo_labels.train_baseline_model_script
```
2. Model Evaluation
```bash
python3 -m pseudo_labels.evaluate_baseline_on_sw_data_script
```
3. Pseudo Labels Algorithm
```bash
python3 -m pseudo_labels.pseudo_labels_script
```
