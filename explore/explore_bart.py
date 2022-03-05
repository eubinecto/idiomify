from transformers import BartTokenizer, BartModel


def main():

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartModel.from_pretrained('facebook/bart-large')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    H_all = outputs.last_hidden_state  # noqa
    print(H_all.shape)  # (1, 8, 1024)


if __name__ == '__main__':
    main()
