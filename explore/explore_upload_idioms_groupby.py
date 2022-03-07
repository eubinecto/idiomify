from idiomify.fetchers import fetch_literal2idiomatic, fetch_config


def main():
    config = fetch_config()['literal2idiomatic']
    train_df, _ = fetch_literal2idiomatic(config['ver'])
    idioms_df = train_df[['Idiom', "Sense"]]
    idioms_df = idioms_df.groupby('Idiom').agg({'Sense': lambda x: list(set(x))})
    print(idioms_df.head(5))
    for idx, row in idioms_df.iterrows():
        print(row['Sense'])

"""
['to arrange something in a manner that either someone will gain a wrong disadvantage or a person would get an unfair advantage']
['Used in general to refer an experience or talent or ability or position, which would be useful or beneficial for a person, his life and his future.']
['to be very easy to see or notice']
[' to reach a logical conclusion']
['to start doing something over from the beginning']
"""

if __name__ == '__main__':
    main()