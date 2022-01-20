import argparse
from idiomify.fetchers import fetch_config, fetch_idioms, fetch_rd
from idiomify import tensors as T
from transformers import BertTokenizer


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str,
                            default="alpha")
        parser.add_argument("--ver", type=str,
                            default="eng2eng")
        parser.add_argument("--sent", type=str,
                            default="avoid getting to the point")
        args = parser.parse_args()
        config = fetch_config()[args.model][args.ver]
        config.update(vars(args))
        tokenizer = BertTokenizer.from_pretrained(config['bert'])
        idioms = fetch_idioms(config['idioms_ver'])
        X = T.inputs([config['sent']], tokenizer, config['k'])
        rd = fetch_rd(config['model'], config['ver'])
        probs = rd.P_wisdom(X).squeeze().tolist()
        wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(idioms, probs)
        ]
        # sort and append
        res = list(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        for idx, (idiom, prob) in enumerate(res):
            print(idx, idiom, prob)


if __name__ == '__main__':
    main()
