import re
from typing import List
from transformers import BartTokenizer
from idiomify.builders import SourcesBuilder
from idiomify.models import Idiomifier


class Pipeline:

    def __init__(self, model: Idiomifier, tokenizer: BartTokenizer):
        self.model = model
        self.builder = SourcesBuilder(tokenizer)

    def __call__(self, sents: str, max_length=300) -> str:
        # yeah... I just want to see what happens here?
        srcs = self.builder(literal2idiomatic=[(sents, "")])
        pred_ids = self.model.bart.generate(
            inputs=srcs[:, 0],  # (N, 2, L) -> (N, L)
            attention_mask=srcs[:, 1],  # (N, 2, L) -> (N, L)
            decoder_start_token_id=self.model.hparams['bos_token_id'],
            max_length=max_length,
        ).squeeze()  # -> (N, L_t) -> (L_t)
        tgts = self.builder.tokenizer.decode(pred_ids, skip_special_tokens=False)
        tgts = re.sub(r"<s>|</s>", "", tgts)
        return tgts
