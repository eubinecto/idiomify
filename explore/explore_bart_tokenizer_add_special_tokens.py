from transformers import BartTokenizer, BartForConditionalGeneration


def main():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    num_added_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": ["<idiom>", "</idiom>"],  # beginning and end of an idiom
    })
    print(num_added_tokens)
    print(tokenizer.additional_special_tokens)  # more special tokens are added here
    # and then you should resize the embedding table of your model
    print(bart.model.shared.weight.shape)  # before
    bart.resize_token_embeddings(len(tokenizer))
    print(bart.model.shared.weight.shape)  # after


if __name__ == '__main__':
    main()

"""
2
['<idiom>', '</idiom>']
torch.Size([50265, 768])
torch.Size([50267, 768])  # you can see that 2 more embedding vectors have been added here.
later, you may want to save the tokenizer after you add the idiom special tokens.
"""
