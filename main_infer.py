"""
This is for just a simple sanity check on the inference.
"""
import argparse
from idiomify.fetchers import fetch_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sent", type=str,
                        default="Just remember that there will always be a hope even when things look hopeless")
    args = parser.parse_args()
    pipeline = fetch_pipeline()
    tgts = pipeline(sents=[args.sent])
    print(args.sent, "\n->", tgts[0])


if __name__ == '__main__':
    main()
