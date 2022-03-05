
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import nltk


sent = "I am really happy with the new job and I mean that with sincere feeling"


def main():
    nltk.download("omw-1.4")
    # this seems legit! I could definitely use this to increase the accuracy of the model
    # for a few idioms (possibly ten, ten very different but frequent idioms)
    aug = naw.ContextualWordEmbsAug()
    augmented = aug.augment(sent, n=10)
    print(augmented)


if __name__ == '__main__':
    main()
