"""
all the functions for building tensors are defined here.
builders must accept device as one of the parameters.
"""
import torch
from typing import List, Tuple
from transformers import BartTokenizer, BertTokenizer


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


class InputsBuilder(TensorBuilder):
    """
    to be used for both training and inference
    """
    def __call__(self, literal2entities: List[Tuple[str, str]]) -> torch.Tensor:
        # just encode the literal sentences
        encodings = self.tokenizer(text=[literal for literal, _ in literal2entities],
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True,
                                   add_special_tokens=True)
        srcs = torch.stack([encodings['input_ids'],
                            encodings['token_type_ids'],
                            encodings['attention_mask']], dim=1)   # (N, 3, L)
        return srcs  # (N, 3, L)


class LabelsBuilder:

    def __call__(self, literal2entities: List[Tuple[str, str]], labels: List[str]) -> torch.Tensor:
        # Surely, the length won't stay the same. You should keep them in sync.
        # How do I pad it? -> wait, but this is what you should do in the preprocessing phase.
        labels = torch.IntTensor([
            [labels.index(entity) for entity in entities.split("|")]
            for _, entities in literal2entities
        ])
        return labels  # (N, L)
