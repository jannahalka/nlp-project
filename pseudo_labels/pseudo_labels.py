
def handle(tokens, preds, word_ids: list[int], probs):
    THRESHOLD = 0.90

    last_word_idx = None
    sent_tokens = []
    sent_labels = []

    for idx, (token, label, word_idx) in enumerate(
        zip(tokens, preds.squeeze().cpu().tolist(), word_ids)
    ):
        is_special_token = word_idx == None
        if is_special_token:
            continue
        # clean off BERT's '##' if it's a continuation
        clean = token[2:] if token.startswith("##") else token
        confidence = probs[0, idx, label].item()

        if word_idx != last_word_idx:
            # new word: start fresh
            sent_tokens.append(clean)
            sent_labels.append(label if confidence >= THRESHOLD else -100)
            last_word_idx = word_idx
        else:
            # continuation: merge into the last token, keep its original label
            sent_tokens[-1] += clean

    return sent_tokens, sent_labels
