from transformers import BartTokenizer
from idiomify.builders import TargetsBuilder

BATCH = [
    ("I could die at any moment", "I could kick the bucket at any moment"),
    ("Speak plainly", "Don't beat around the bush")
]


def main():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    builder = TargetsBuilder(tokenizer)
    tgt_r, tgt = builder(BATCH)
    print(tgt_r)
    print(tgt)


if __name__ == '__main__':
    main()
