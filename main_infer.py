"""
This is for just a simple sanity check on the inference.
"""
import argparse
from idiomify.pipeline import Pipeline
from idiomify.fetchers import fetch_config, fetch_idiomifier, fetch_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sents", type=str,
                        default="Just remember that there will always be a hope even when things look hopeless")
    args = parser.parse_args()
    config = fetch_config()['idiomifier']
    config.update(vars(args))
    model = fetch_idiomifier(config['ver'])
    tokenizer = fetch_tokenizer(config['tokenizer_ver'])
    model.eval()  # this is crucial
    pipeline = Pipeline(model, tokenizer)
    tgts = pipeline(sents=config['sents'])
    print(config['sents'], "\n->", tgts)


if __name__ == '__main__':
    main()
