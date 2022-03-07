
from idiomify.fetchers import fetch_pie
from preprocess import annotate


def main():
    pie_df = fetch_pie()
    pie_df = pie_df.pipe(annotate, boi_token="<idiom>", eoi_token="</idiom>")
    for _, row in pie_df.iterrows():
        print(row['Idiomatic_Sent'])


if __name__ == '__main__':
    main()
