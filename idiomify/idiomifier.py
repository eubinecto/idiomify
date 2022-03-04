from transformers import BartTokenizer
from builders import SourcesBuilder
from models import Alpha


class Idiomifier:

    def __init__(self, model: Alpha, tokenizer: BartTokenizer):
        self.model = model
        self.builder = SourcesBuilder(tokenizer)
        self.model.eval()

    def __call__(self, src: str, max_length=100) -> str:
        srcs = self.builder(literal2idiomatic=[(src, "")])
        pred_ids = self.model.bart.generate(
            inputs=srcs[:, 0],  # (N, 2, L) -> (N, L)
            attention_mask=srcs[:, 1],  # (N, 2, L) -> (N, L)
            decoder_start_token_id=self.model.hparams['bos_token_id'],
            max_length=max_length,
        ).squeeze()  # -> (N, L_t) -> (L_t)
        tgt = self.builder.tokenizer.decode(pred_ids, skip_special_tokens=True)
        return tgt
