
from torchmetrics import BLEUScore
from transformers import BartTokenizer


pairs = [
    ("I knew you could do it", "I knew you could do it"),
    ("I knew you could do it", "you knew you could do it")
]


def main():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    metric = BLEUScore()
    preds = tokenizer([pred for pred, _ in pairs])['input_ids']
    targets = tokenizer([target for _, target in pairs])['input_ids']
    print(preds)
    print(targets)
    print(metric(preds, targets))
    # arghhh, so bleu score does not support tensors...
    """
    AttributeError: 'int' object has no attribute 'split'
    """
    # let's just go for the accuracies then.


if __name__ == '__main__':
    main()
