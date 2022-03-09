
import nlpaug.augmenter.word as naw
import nltk


sent = "I am really happy with the new job and I mean that with sincere feeling"


def main():
    nltk.download("omw-1.4")
    # this seems legit! I could definitely use this to increase the accuracy of the model
    # for a few idioms (possibly ten, ten very different but frequent idioms)
    aug = naw.ContextualWordEmbsAug()
    augmented = aug.augment(sent, n=10)
    print(augmented)
    for var in augmented:
        # Does the length stay the same?
        # Oh yes it does!
        # this is great, as this could be used for augmenting data even for
        # ner dataset.
        print(var)
        print(len(var.split(" ")))


if __name__ == '__main__':
    main()
