"""
all the functions for building tensors are defined here.
builders must accept device as one of the parameters.
"""
import torch
from typing import List, Tuple
from transformers import BertTokenizer


class TensorBuilder:

    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class Idiom2SubwordsBuilder(TensorBuilder):

    def __call__(self, idioms: List[str], k: int) -> torch.Tensor:
        """
                1. The function takes in a list of idioms, and a maximum length of the input sequence.
                2. It then splits the idioms into words, and pads the sequence to the maximum length.
                3. It masks the padding tokens, and returns the input ids
                :param idioms: a list of idioms, each of which is a list of tokens
                :type idioms: List[str]
                :param k: the maximum length of the idioms
                :type k: int
                :return: The input_ids of the idioms, with the pad tokens replaced by the mask token.
        """
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        # temporarily disable single-token status of the idioms
        idioms = [idiom.split(" ") for idiom in idioms]
        encodings = self.tokenizer(text=idioms,
                                   add_special_tokens=False,
                                   # should set this to True, as we already have the idioms split.
                                   is_split_into_words=True,
                                   padding='max_length',
                                   max_length=k,  # set to k
                                   return_tensors="pt")
        input_ids = encodings['input_ids']
        input_ids[input_ids == pad_id] = mask_id
        return input_ids


class Idiom2ContextBuilder(TensorBuilder):

    def __call__(self, idiom2context: List[Tuple[str, str]]):
        """
            Given a list of tuples of idiom and context,
            it returns a tensor of shape (batch_size, 3, max_seq_len)
            :param idiom2context: List[Tuple[str, str]], a list of tuples of idiom and context
            :type idiom2context: List[Tuple[str, str]]
            :return: The input_ids, token_type_ids, and attention_mask for each context.
        """
        contexts = [context for _, context in idiom2context]
        encodings = self.tokenizer(text=contexts,
                                   return_tensors="pt",
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   verbose=True)
        return torch.stack([encodings['input_ids'],
                            encodings['token_type_ids'],
                            encodings['attention_mask']], dim=1)


class TargetsBuilder(TensorBuilder):

    def __call__(self, idiom2sent: List[Tuple[str, str]], idioms: List[str]) -> torch.Tensor:
        """
            Given a list of idioms and a list of sentences, return a list of indices of the idioms in the sentences
            :param idiom2sent: A list of tuples, where each tuple is an idiom and its corresponding sentence
            :type idiom2sent: List[Tuple[str, str]]
            :param idioms: A list of idioms
            :type idioms: List[str]
            :return: A tensor of indices of the idioms in the list of idioms.
        """
        return torch.LongTensor([
            idioms.index(idiom)
            for idiom, _ in idiom2sent
        ])
