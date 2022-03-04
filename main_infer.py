import argparse
from termcolor import colored
from idiomifier import Idiomifier
from idiomify.fetchers import fetch_config, fetch_alpha
from transformers import BartTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="alpha")
    parser.add_argument("--ver", type=str,
                        default="overfit")
    parser.add_argument("--src", type=str,
                        default="If there's any benefits to losing my job, it's that I'll now be able to go to school full-time and finish my degree earlier.")
    args = parser.parse_args()
    config = fetch_config()[args.model][args.ver]
    config.update(vars(args))
    model = fetch_alpha(config['ver'])
    tokenizer = BartTokenizer.from_pretrained(config['bart'])
    idiomifier = Idiomifier(model, tokenizer)
    src = config['src']
    tgt = idiomifier(src=config['src'])
    print(src, "\n->", colored(tgt, "blue"))


if __name__ == '__main__':
    main()
