
from idiomify.fetchers import fetch_pie


def main():
    for idx, row in enumerate(fetch_pie()):
        print(idx, row)
        # the first 105 = V0.
        if idx == 105:
            break


if __name__ == '__main__':
    main()