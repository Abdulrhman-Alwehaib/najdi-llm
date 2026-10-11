from najdi.training.templates import build_examples, choose_pad, end_of_turn


def test_end_of_turn_and_pad(tokenizer):
    eot = end_of_turn(tokenizer)
    assert eot == "<|im_end|>"
    assert choose_pad(tokenizer, eot) == "<pad>"


def test_every_reply_is_trained_with_its_end_of_turn(tokenizer):
    eot = end_of_turn(tokenizer)
    eot_id = tokenizer.convert_tokens_to_ids(eot)
    messages = [
        {"role": "user", "content": "هلا"},
        {"role": "assistant", "content": "هلا بك"},
        {"role": "user", "content": "وش اخبارك"},
        {"role": "assistant", "content": "بخير"},
    ]
    (example,) = build_examples(tokenizer, messages, eot)
    assert len(example["input_ids"]) == len(example["labels"])
    trained = [i for i, label in enumerate(example["labels"]) if label != -100]
    assert example["labels"][-1] == eot_id
    assert sum(1 for i in trained if example["labels"][i] == eot_id) == 2
    unscored = build_examples(tokenizer, messages, eot, score_eot=False)[0]
    assert eot_id not in [label for label in unscored["labels"] if label != -100]
