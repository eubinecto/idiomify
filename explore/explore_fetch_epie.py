
from idiomify.fetchers import fetch_epie


def main():
    epie = fetch_epie()
    idioms = set([
        idiom
        for idiom, _, _ in epie
    ])

    # so, what do you want? you want to build an idiom-masked language modeling?
    for idiom, context, tag in epie:
        print(idiom, context)

    for idx, idiom in enumerate(idioms):
        print(idx, idiom)

    # isn't it better to just leave the idiom there, and have it guess what meaning it has?
    # in that case, It may be better to use a generative model?
    # but what would happen if you let it... just guess it?
    # the problem with non-masking is that ... you give the model the answer.
    # what you should rather do is... do something like...  find similar words.


if __name__ == '__main__':
    main()