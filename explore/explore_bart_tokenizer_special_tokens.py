from transformers import BartTokenizer


def main():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    print(tokenizer.bos_token)
    print(tokenizer.cls_token)
    print(tokenizer.eos_token)
    print(tokenizer.sep_token)
    print(tokenizer.mask_token)
    print(tokenizer.pad_token)
    print(tokenizer.unk_token)


"""
<s>
<s>
</s>
</s>
<mask>
<pad>
<unk>

right, so this is just like the symbols for BERT but in lowercase. 
bos = cls 
sep = eos
would it be okay to use <idiom> = <sep>? 
no, sep implies that a sentence somehow ends. 
"""





if __name__ == '__main__':
    main()
