from idiomify.fetchers import fetch_idiom2def
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast


def main():

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    idiom2def = fetch_idiom2def("c")  # eng2eng
    for idiom, definition in idiom2def:
        print(tokenizer.decode(tokenizer(idiom)['input_ids']),
              tokenizer.decode(tokenizer(definition)['input_ids']))

"""
/opt/homebrew/Caskroom/miniforge/base/envs/idiomify-demo/bin/python /Users/eubinecto/Desktop/Projects/Toy/idiomify-demo/explore/explore_bert_base_tokenizer.py
Downloading: 100%|██████████| 226k/226k [00:00<00:00, 298kB/s]
Downloading: 100%|██████████| 28.0/28.0 [00:00<00:00, 8.27kB/s]
Downloading: 100%|██████████| 455k/455k [00:01<00:00, 449kB/s]
[CLS] beat around the bush [SEP] [CLS] to speak vaguely or euphemistically so as to avoid talkingdirectly about an unpleasant or sensitive topic [SEP]
[CLS] beat around the bush [SEP] [CLS] indirection in word or deed [SEP]
[CLS] beat around the bush [SEP] [CLS] to shilly - shally [SEP]
[CLS] beat around the bush [SEP] [CLS] to approach something in a roundabout way [SEP]
[CLS] backhanded compliment [SEP] [CLS] an insulting or negative comment disguised as praise. [SEP]
[CLS] backhanded compliment [SEP] [CLS] an unintended or ambiguous compliment. [SEP]
[CLS] backhanded compliment [SEP] [CLS] a remark which seems to be praising someone or something but which could also be understood as criticism [SEP]
[CLS] backhanded compliment [SEP] [CLS] a remark that seems to say something pleasant about a person but could also be an insult [SEP]
[CLS] backhanded compliment [SEP] [CLS] a remark that seems to express admiration but could also be understood as an insult [SEP]
[CLS] steer clear of [SEP] [CLS] to avoid someone or something. [SEP]
[CLS] steer clear of [SEP] [CLS] stay away from [SEP]
[CLS] steer clear of [SEP] [CLS] take care to avoid or keep away from [SEP]
[CLS] steer clear of [SEP] [CLS] to avoid someone or something that seems unpleasant, dangerous, or likely to cause problems [SEP]
[CLS] steer clear of [SEP] [CLS] deliberately avoid someone [SEP]
[CLS] dish it out [SEP] [CLS] to voice harsh thoughts, criticisms, or insults. [SEP]
[CLS] dish it out [SEP] [CLS] to gossip about someone or something [SEP]
[CLS] dish it out [SEP] [CLS] to give something, or to tell something such as information or your opinions [SEP]
[CLS] dish it out [SEP] [CLS] someone easily criticizes other people but does not like it when other people criticize him or her [SEP]
[CLS] dish it out [SEP] [CLS] to criticize other people [SEP]
[CLS] make headway [SEP] [CLS] make progress with something that you are trying to achieve. [SEP]
[CLS] make headway [SEP] [CLS] make progress, especially when this is slow or difficult [SEP]
[CLS] make headway [SEP] [CLS] to advance. [SEP]
[CLS] make headway [SEP] [CLS] to move forward or make progress [SEP]
[CLS] make headway [SEP] [CLS] to begin to succeed [SEP]
"""

if __name__ == '__main__':
    main()
