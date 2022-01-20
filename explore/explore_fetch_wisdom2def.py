from idiomify.fetchers import fetch_wisdom2def


def main():
    df = fetch_wisdom2def("c")
    for idx, row in df.iterrows():
        print(row[0], row[1])

    df = fetch_wisdom2def("d")
    for idx, row in df.iterrows():
        print(row[0], row[1])


if __name__ == '__main__':
    main()
