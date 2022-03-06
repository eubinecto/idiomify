from typing import List
from transformers import BartTokenizer
from idiomify.builders import SourcesBuilder
from idiomify.models import Idiomifier


class Pipeline:

    def __init__(self, model: Idiomifier, tokenizer: BartTokenizer):
        self.model = model
        self.builder = SourcesBuilder(tokenizer)

    def __call__(self, sents: List[str], max_length=100) -> List[str]:
        srcs = self.builder(literal2idiomatic=[(sent, "") for sent in sents])
        pred_ids = self.model.bart.generate(
            inputs=srcs[:, 0],  # (N, 2, L) -> (N, L)
            attention_mask=srcs[:, 1],  # (N, 2, L) -> (N, L)
            decoder_start_token_id=self.model.hparams['bos_token_id'],
            max_length=max_length,
        )  # -> (N, L_t)
        tgts = self.builder.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        return tgts
