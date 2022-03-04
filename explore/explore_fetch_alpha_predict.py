from transformers import BartTokenizer
from builders import SourcesBuilder
from fetchers import fetch_alpha


def main():
    model = fetch_alpha("overfit")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    lit2idi = [
        ("my man", ""),
        ("hello", "")
    ]  # just some dummy stuff
    srcs = SourcesBuilder(tokenizer)(lit2idi)
    out = model.predict(srcs=srcs)
    print(out)


if __name__ == '__main__':
    main()
