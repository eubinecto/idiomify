from idiomify.fetchers import fetch_literal2idiomatic


def main():
    train_df, test_df = fetch_literal2idiomatic("d-1-2")
    print(train_df.size)  # 12408 rows
    print(test_df.size)  # 3102 rows


if __name__ == '__main__':
    main()
