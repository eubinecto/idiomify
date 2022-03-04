from idiomify.fetchers import fetch_alpha


def main():
    model = fetch_alpha("overfit")
    print(model.bart.config)


if __name__ == '__main__':
    main()
