from idiomify.fetchers import fetch_idiom2def


def main():
    idiom2def = fetch_idiom2def("c")
    for idiom, definition in idiom2def:
        print(idiom, definition)

    df = fetch_idiom2def("d")
    for idiom, definition in idiom2def:
        print(idiom, definition)


if __name__ == '__main__':
    main()
