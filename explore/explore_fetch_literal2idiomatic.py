from idiomify.fetchers import fetch_literal2idiomatic


def main():
    for src, tgt in fetch_literal2idiomatic("pie_v0"):
        print(src, "->", tgt)


if __name__ == '__main__':
    main()
