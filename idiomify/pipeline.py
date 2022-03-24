import re
import pandas as pd
from typing import List
from transformers import BartTokenizer
from idiomify.builders import InputsBuilder
from idiomify.models import Idiomifier


class Pipeline:

    def __init__(self, model: Idiomifier, tokenizer: BartTokenizer, idioms: pd.DataFrame):
        self.model = model
        self.builder = InputsBuilder(tokenizer)
        self.idioms = idioms

    def __call__(self, sents: List[str], max_length=100) -> List[str]:
        srcs = self.builder(literal2idiomatic=[(sent, "") for sent in sents])
        pred_ids = self.model.bart.generate(
            inputs=srcs[:, 0],  # (N, 2, L) -> (N, L)
            attention_mask=srcs[:, 1],  # (N, 2, L) -> (N, L)
            decoder_start_token_id=self.model.hparams['bos_token_id'],
            max_length=max_length,
        )  # -> (N, L_t)
        # we don't skip special tokens because we have to keep <idiom> & </idiom> for highlighting idioms.
        tgts = self.builder.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
        tgts = [
            re.sub(r"<s>|</s>|<pad>", "", tgt)
            for tgt in tgts
        ]
        return tgts
