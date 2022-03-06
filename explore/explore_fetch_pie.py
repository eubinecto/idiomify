
from idiomify.fetchers import fetch_pie


def main():
    pie_df = fetch_pie()
    for idx, row in pie_df.iterrows():
        print(row)


if __name__ == '__main__':
    main()