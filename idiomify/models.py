"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from typing import Tuple, List, Optional
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import BertForMaskedLM


class Idiomifier(pl.LightningModule):
    """
    @eubinecto
    The superclass of all the reverse-dictionaries. This class houses any methods that are required by
    whatever reverse-dictionaries we define.
    """
    # passing them to avoid warnings ---  #
    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def __init__(self, mlm: BertForMaskedLM, idiom2subwords: torch.Tensor, k: int, lr: float):  # noqa
        """
        :param mlm: a bert model for masked language modeling
        :param idiom2subwords: (|W|, K)
        :return: (N, K, |V|); (num samples, k, the size of the vocabulary of subwords)
        """
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        given a batch, forward returns a batch of hidden vectors
        :param X: (N, 3, L). input_ids, token_type_ids, and what was the last one...?
        :return: (N, L, H)
        """
        pass

    def step(self):
        pass

    def predict(self):
        pass

    def training_step(self):
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it as well.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower()


class Alpha(Idiomifier):
    """
    @eubinecto
    The first prototype.
    S_wisdom = S_wisdom_literal
    trained on: wisdom2def only.
    """

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        H_k = self.H_k(H_all)  # (N, L, H) -> (N, K, H)
        S_wisdom = self.S_wisdom_literal(H_k)  # (N, K, H) -> (N, |W|)
        return S_wisdom
