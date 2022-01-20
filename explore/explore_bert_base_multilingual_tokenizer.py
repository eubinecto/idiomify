from idiomify.fetchers import fetch_idiom2def
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    idiom2def = fetch_idiom2def("d")  # eng2kor

    for idiom, definition in idiom2def:
        print(tokenizer.decode(tokenizer(idiom)['input_ids']),
              tokenizer.decode(tokenizer(definition)['input_ids']))

# right, the tokenizer knows Korean, which is great.
"""
/opt/homebrew/Caskroom/miniforge/base/envs/idiomify-demo/bin/python /Users/eubinecto/Desktop/Projects/Toy/idiomify-demo/explore/explore_mbert_tokenizer.py
[CLS] beat around the bush [SEP] [CLS] 불쾌하거나 민감한 주제에 대해 직접적으로 이야기하는 것을 피하기 위해 모호하거나 완곡하게 말한다. [SEP]
[CLS] beat around the bush [SEP] [CLS] 단어나 태도가 우회적이다 [SEP]
[CLS] beat around the bush [SEP] [CLS] 우물쭈물하다 [SEP]
[CLS] beat around the bush [SEP] [CLS] 우회적으로 접근하다 [SEP]
[CLS] backhanded compliment [SEP] [CLS] 칭찬으로 가장한 모욕적이거나 부정적인 논평 [SEP]
[CLS] backhanded compliment [SEP] [CLS] 의도하지 않거나 애매한 칭찬 [SEP]
[CLS] backhanded compliment [SEP] [CLS] 누군가를 칭찬하는 것 같지만 비판으로도 이해될 수 있는 말 [SEP]
[CLS] backhanded compliment [SEP] [CLS] 남을 기쁘게 하는 말 같지만 모욕이 될 수도 있는 말 [SEP]
[CLS] backhanded compliment [SEP] [CLS] 감탄하는 듯 하면서도 모욕으로 이해될 수 있는 말 [SEP]
[CLS] steer clear of [SEP] [CLS] 누군가나 뭔가를 피하다 [SEP]
[CLS] steer clear of [SEP] [CLS] 떨어져 지내다 [SEP]
[CLS] steer clear of [SEP] [CLS] 피하거나 멀리하도록 주의하다 [SEP]
[CLS] steer clear of [SEP] [CLS] 불쾌하거나 위험하거나 문제를 일으킬 것 같은 사람이나 물건을 피하다 [SEP]
[CLS] steer clear of [SEP] [CLS] 일부러 피하다 [SEP]
[CLS] dish it out [SEP] [CLS] 가혹한 생각, 비판, 또는 모욕의 목소리를 내는 것. [SEP]
[CLS] dish it out [SEP] [CLS] 누군가 또는 무언가에 대해 험담하는 것 [SEP]
[CLS] dish it out [SEP] [CLS] 어떤 것을 주거나 정보나 당신의 의견과 같은 것을 말하는 것 [SEP]
[CLS] dish it out [SEP] [CLS] 다른 사람을 쉽게 비판하지만 다른 사람이 자신을 비판할때는 좋아하지 않음 [SEP]
[CLS] dish it out [SEP] [CLS] 다른 사람을 비판하다 [SEP]
[CLS] make headway [SEP] [CLS] 성취하고자 하는 어떤 것에 진척이 생기다 [SEP]
[CLS] make headway [SEP] [CLS] 특히 이것이 느리거나 어려울 때, 진전을 이루다. [SEP]
[CLS] make headway [SEP] [CLS] 전진하다 [SEP]
[CLS] make headway [SEP] [CLS] 앞으로 나아가거나 진전을 이루다 [SEP]
[CLS] make headway [SEP] [CLS] 성공하기 시작하다 [SEP]
"""


if __name__ == '__main__':
    main()
