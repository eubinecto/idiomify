
from idiomify.fetchers import fetch_pie
from preprocess import replace_labels


def main():
    df = fetch_pie()
    df = df.pipe(replace_labels)
    for _, row in df.iterrows():
        print(row['Literal_Label'])

"""
seems about right!
['B/fast food', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I/fast food']
['O', 'O', 'O', 'B/fast food', 'I/fast food', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'B/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'B/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['B/catch up', 'I/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'B/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'O', 'B/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'O', 'B/catch up', 'I/catch up', 'I/catch up', 'I/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B/catch up', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'B/catch up', 'I/catch up', 'I/catch up', 'O', 'O']
['O', 'O', 'O', 'O', 'B/catch up', 'I/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'B/catch up', 'I/catch up', 'I/catch up', 'I/catch up', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'B/catch up', 'I/catch up', 'I/catch up', 'I/catch up', 'O']
['B/catch up', 'I/catch up', 'I/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'B/catch up', 'I/catch up', 'I/catch up', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B/catch up', 'I/catch up', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B/catch up', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'B/keep an eye on', 'I/keep an eye on', 'I/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'B/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'B/keep an eye on', 'O', 'I/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'B/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'I/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'B/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'B/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['B/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'O', 'B/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'B/keep an eye on', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B/keep an eye on', 'I/keep an eye on', 'O', 'O', 'O', 'O']
"""


if __name__ == '__main__':
    main()
