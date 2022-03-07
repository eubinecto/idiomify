from idiomify.fetchers import fetch_tokenizer


def main():
    tokenizer = fetch_tokenizer("t-1-1")
    sent = "There will always be a <idiom> silver lining </idiom> even when things look pitch black"
    ids = tokenizer(sent)['input_ids']
    print(ids)
    decoded = tokenizer.decode(ids)
    print(decoded)


if __name__ == '__main__':
    main()
