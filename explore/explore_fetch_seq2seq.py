from idiomify.fetchers import fetch_seq2seq


def main():
    model = fetch_seq2seq("overfit")
    print(model.bart.config)


if __name__ == '__main__':
    main()
