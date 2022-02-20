
from idiomify.fetchers import fetch_epie


def main():
    idioms = set([
        idiom
        for idiom, _, _ in fetch_epie()
    ])
    contexts = [
        context
        for _, _, context in fetch_epie()
    ]
    print("Total number of idioms:", len(idioms))
    # This should learn... this - what I need for now is building a datamodule out of this
    print("Total number of contexts:", len(contexts))


if __name__ == '__main__':
    main()
