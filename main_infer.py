import argparse
from idiomify.models import Pipeline
from idiomify.fetchers import fetch_config, fetch_idiomifier
from transformers import BartTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str,
                        default="If there's any good to loosing my job,"
                                " it's that I'll now be able to go to school full-time and finish my degree earlier.")
    args = parser.parse_args()
    config = fetch_config()['idiomifier']
    config.update(vars(args))
    model = fetch_idiomifier(config['ver'])
    model.eval()  # this is crucial
    tokenizer = BartTokenizer.from_pretrained(config['bart'])
    pipeline = Pipeline(model, tokenizer)
    src = config['src']
    tgt = pipeline(src=config['src'])
    print(src, "\n->", tgt)


if __name__ == '__main__':
    main()
