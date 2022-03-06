from idiomify.fetchers import fetch_idiomifier


def main():
    model = fetch_idiomifier("m-1-2")
    print(model.bart.config)


if __name__ == '__main__':
    main()
