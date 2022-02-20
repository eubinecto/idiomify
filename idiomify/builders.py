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
        input_ids[input_ids == pad_id] = mask_id  # replace them with masks
        return input_ids


class Idiom2DefBuilder(TensorBuilder):

    def __call__(self, idiom2def: List[Tuple[str, str]], k: int) -> torch.Tensor:
        defs = [definition for _, definition in idiom2def]
        lefts = [" ".join(["[MASK]"] * k)] * len(defs)
        encodings = self.tokenizer(text=lefts,
                                   text_pair=defs,
                                   return_tensors="pt",
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   verbose=True)
        input_ids: torch.Tensor = encodings['input_ids']
        cls_id: int = self.tokenizer.cls_token_id
        sep_id: int = self.tokenizer.sep_token_id
        mask_id: int = self.tokenizer.mask_token_id
        wisdom_mask = torch.where(input_ids == mask_id, 1, 0)
        desc_mask = torch.where(((input_ids != cls_id) & (input_ids != sep_id) & (input_ids != mask_id)), 1, 0)
        return torch.stack([input_ids,
                            encodings['token_type_ids'],
                            encodings['attention_mask'],
                            wisdom_mask,
                            desc_mask], dim=1)


class Idiom2ContextBuilder(TensorBuilder):

    def __call__(self, idiom2context: List[Tuple[str, str]]):
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
        return torch.LongTensor([
            idioms.index(idiom)
            for idiom, _ in idiom2sent
        ])
