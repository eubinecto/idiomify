"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from typing import Tuple
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartTokenizer
from idiomify.builders import SourcesBuilder


class Idiomifier(pl.LightningModule):  # noqa
    """
    the baseline is in here.
    """
    def __init__(self, bart: BartForConditionalGeneration, lr: float, bos_token_id: int, pad_token_id: int):  # noqa
        super().__init__()
        self.bart = bart
        self.save_hyperparameters(ignore=["bart"])

    def forward(self, srcs: torch.Tensor, tgts_r: torch.Tensor) -> torch.Tensor:
        """
        as for using bart for CG, refer to:
        https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForQuestionAnswering.forward
        param srcs: (N, 2, L_s)
        param tgts_r: (N, 2, L_t)
        return: (N, L, |V|)
        """
        input_ids, attention_mask = srcs[:, 0], srcs[:, 1]
        decoder_input_ids, decoder_attention_mask = tgts_r[:, 0], tgts_r[:, 1]
        outputs = self.bart(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask)
        logits = outputs[0]  # (N, L, |V|)
        return logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict:
        srcs, tgts_r, tgts = batch  # (N, 2, L_s), (N, 2, L_t), (N, 2, L_t)
        logits = self.forward(srcs, tgts_r).transpose(1, 2)  # ... -> (N, L, |V|) -> (N, |V|, L)
        loss = F.cross_entropy(logits, tgts, ignore_index=self.hparams['pad_token_id'])\
                .sum()  # (N, L, |V|), (N, L) -> (N,) -> (1,)
        return {
            "loss": loss
        }

    def on_train_batch_end(self, outputs: dict, *args, **kwargs):
        self.log("Train/Loss", outputs['loss'])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it as well.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


# for inference
class Pipeline:

    def __init__(self, model: Idiomifier, tokenizer: BartTokenizer):
        self.model = model
        self.builder = SourcesBuilder(tokenizer)

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
