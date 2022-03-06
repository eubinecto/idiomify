from fetchers import fetch_pie


def main():
    pie_df = fetch_pie()
    print(pie_df.columns)
    pie_df = pie_df[["Literal_Sent", "Idiomatic_Sent"]]
    print(pie_df.head(5))


if __name__ == '__main__':
    main()
