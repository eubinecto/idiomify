
from idiomify.fetchers import fetch_pie


def main():
    pie_df = fetch_pie()
    for idx, row in pie_df.iterrows():
        print("---")
        print(row['Idiom'])
        print([(token, label) for token, label in zip(row['Literal_Sent'].split(" "), row['Literal_Label'].split(" "))])


if __name__ == '__main__':
    main()