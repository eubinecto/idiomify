"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from typing import Tuple
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertForTokenClassification(BertPreTrainedModel):
    """
    defining a custom head for a pre-trained BERT
    """
    config_class = BertConfig

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # load & init weights
        self.init_weights()

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor)\
            -> TokenClassifierOutput:
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = ...  # (N, L, H)
        loss = ...  # (N,)
        logits = ...  # (N, L, |V|)
        attentions = ...  # what is  the dimension of this? the multi-head dimensions?
        return TokenClassifierOutput(loss=loss, logits=logits,
                                     hidden_states=hidden_states, attentions=attentions)

    # and then what? what should we do further from this?

class Idiomifier(pl.LightningModule):  # noqa
    """
    the baseline is in here.
    """
    def __init__(self, bert: BertModel, lr: float):  # noqa
        super().__init__()
        self.save_hyperparameters(ignore=["bert"])
        self.bert = bert
        self.cls = torch.nn.Linear(..., ...)  # the token-level classifier
        # metrics (using accuracies as of right now)
        self.acc_train = Accuracy()
        self.acc_test = Accuracy()

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
        self.acc_train.update(logits.detach(), target=tgts.detach())
        return {
            "loss": loss
        }

    def on_train_batch_end(self, outputs: dict, *args, **kwargs):
        self.log("Train/Loss", outputs['loss'])

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        self.log("Train/Accuracy", self.acc_train.compute())
        self.acc_train.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args, **kwargs):
        srcs, tgts_r, tgts = batch  # (N, 2, L_s), (N, 2, L_t), (N, 2, L_t)
        logits = self.forward(srcs, tgts_r).transpose(1, 2)  # ... -> (N, L, |V|) -> (N, |V|, L)
        self.acc_test.update(logits.detach(), target=tgts.detach())

    def on_test_epoch_end(self, *args, **kwargs) -> None:
        self.log("Test/Accuracy", self.acc_test.compute())
        self.acc_test.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it as well.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])
