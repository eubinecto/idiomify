from idiomify.fetchers import fetch_tokenizer


def main():
    tokenizer = fetch_tokenizer("t-1-1")
    print(tokenizer.bos_token)
    print(tokenizer.cls_token)
    print(tokenizer.eos_token)
    print(tokenizer.sep_token)
    print(tokenizer.mask_token)
    print(tokenizer.pad_token)
    print(tokenizer.unk_token)
    print(tokenizer.additional_special_tokens)  # this should have been added

    # the size of the vocab
    print(len(tokenizer))


"""
<s>
<s>
</s>
</s>
<mask>
<pad>
<unk>
['<idiom>', '</idiom>']
50267
"""

if __name__ == '__main__':
    main()
