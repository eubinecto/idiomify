import argparse
from idiomify.models import Idiomifier
from idiomify.fetchers import fetch_config, fetch_seq2seq
from transformers import BartTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="tag011")
    parser.add_argument("--src", type=str,
                        default="If there's any good to loosing my job,"
                                " it's that I'll now be able to go to school full-time and finish my degree earlier.")
    args = parser.parse_args()
    config = fetch_config()[args.ver]
    config.update(vars(args))
    model = fetch_seq2seq(config['ver'])
    tokenizer = BartTokenizer.from_pretrained(config['bart'])
    idiomifier = Idiomifier(model, tokenizer)
    src = config['src']
    tgt = idiomifier(src=config['src'])
    print(src, "\n->", tgt)


if __name__ == '__main__':
    main()
