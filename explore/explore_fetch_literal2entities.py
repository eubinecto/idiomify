from idiomify.fetchers import fetch_literal2entities


def main():
    train_df, test_df = fetch_literal2entities("d-1-4")
    print(train_df.size)  # 12408 rows
    print(test_df.size)  # 3102 rows


if __name__ == '__main__':
    main()
