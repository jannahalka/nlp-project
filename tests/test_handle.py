from models.baseline_model.predict import handle

input_ids = None
tokenizer = None
tokens = None
probs = None
preds = None
word_ids = None



def test_handle():
    handle(tokens, preds, word_ids, probs)
