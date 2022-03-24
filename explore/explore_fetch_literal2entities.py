from idiomify.fetchers import fetch_literal2entities


def main():
    train_df, test_df = fetch_literal2entities("d-1-4")
    # the size
    print(train_df.size)  # 12408 rows
    print(test_df.size)  # 3102 rows
    # some entries
    print(train_df.head())
    print(train_df.columns)


if __name__ == '__main__':
    main()
