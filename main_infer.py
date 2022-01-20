import argparse
from idiomify import tensors as T
from idiomify.fetchers import fetch_config, fetch_rd, fetch_idioms
from transformers import BertTokenizer


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str,
                            default="alpha")
        parser.add_argument("--ver", type=str,
                            default="eng2eng")
        parser.add_argument("--sent", type=str,
                            default="to avoid getting to the point")
        args = parser.parse_args()
        config = fetch_config()[args.model][args.ver]
        config.update(vars(args))
        idioms = fetch_idioms(config['idioms_ver'])
        rd = fetch_rd(config['model'], config['ver'])
        rd.eval()
        tokenizer = BertTokenizer.from_pretrained(config['bert'])
        X = T.inputs([("", config['sent'])], tokenizer, config['k'])
        probs = rd.P_wisdom(X).squeeze().tolist()
        wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(idioms, probs)
        ]
        # sort and append
        res = list(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        print(f"query: {config['sent']}")
        for idx, (idiom, prob) in enumerate(res):
            print(idx, idiom, prob)


if __name__ == '__main__':
    main()
