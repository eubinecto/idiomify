from idiomify.fetchers import fetch_idioms


def main():
    df = fetch_idioms("d-1-4")
    for idx, row in df.iterrows():
        print(row[0])


if __name__ == '__main__':
    main()
