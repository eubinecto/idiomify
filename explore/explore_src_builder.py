from transformers import BartTokenizer
from idiomify.builders import SourcesBuilder

BATCH = [
    ("I could die at any moment", "I could kick the bucket at any moment"),
    ("Speak plainly", "Don't beat around the bush")
]


def main():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    builder = SourcesBuilder(tokenizer)
    src = builder(BATCH)
    print(src)


if __name__ == '__main__':
    main()
