"""
all the functions for building tensors are defined here.
builders must accept device as one of the parameters.
"""
import torch
from typing import List, Tuple
from transformers import BartTokenizer


class TensorBuilder:

    def __init__(self, tokenizer: BartTokenizer):
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


class SourcesBuilder(TensorBuilder):
    """
    to be used for both training and inference
    """
    def __call__(self, literal2idiomatic: List[Tuple[str, str]]) -> torch.Tensor:
        encodings = self.tokenizer(text=[literal for literal, _ in literal2idiomatic],
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True,
                                   add_special_tokens=True)
        srcs = torch.stack([encodings['input_ids'],
                            encodings['attention_mask']], dim=1)   # (N, 2, L)
        return srcs  # (N, 2, L)


class TargetsRightShiftedBuilder(TensorBuilder):
    """
    This is to be used only for training. As for inference, we don't need this.
    """
    def __call__(self, literal2idiomatic: List[Tuple[str, str]]) -> torch.Tensor:
        encodings = self.tokenizer([
            self.tokenizer.bos_token + idiomatic  # starts with bos, but does not end with eos (right-shifted)
            for _, idiomatic in literal2idiomatic
        ], return_tensors="pt", add_special_tokens=False, padding=True, truncation=True)
        tgts_r = torch.stack([encodings['input_ids'],
                              encodings['attention_mask']], dim=1)  # (N, 2, L)
        return tgts_r


class TargetsBuilder(TensorBuilder):

    def __call__(self, literal2idiomatic: List[Tuple[str, str]]) -> torch.Tensor:
        encodings = self.tokenizer([
            idiomatic + self.tokenizer.eos_token  # no bos, but ends with eos
            for _, idiomatic in literal2idiomatic
        ], return_tensors="pt", add_special_tokens=False, padding=True, truncation=True)
        tgts = encodings['input_ids']
        return tgts  # (N, L)


