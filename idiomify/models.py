"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from typing import Tuple, List, Optional
import torch
import pytorch_lightning as pl
from transformers.models.bert.modeling_bert import BertForMaskedLM
from torch.nn import functional as F


class RD(pl.LightningModule):
    """
    @eubinecto
    The superclass of all the reverse-dictionaries. This class houses any methods that are required by
    whatever reverse-dictionaries we define.
    """

    # --- boilerplate; the loaders are defined in datamodules, so we don't define them here
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
        super().__init__()
        # -- hyper params --- #
        # should be saved to self.hparams
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4390#issue-730493746
        self.save_hyperparameters(ignore=["mlm", "idiom2subwords"])
        # -- the only neural network we need -- #
        self.mlm = mlm
        # --- to be used for getting H_k --- #
        self.wisdom_mask: Optional[torch.Tensor] = None  # (N, L)
        # --- to be used for getting H_desc --- #
        self.desc_mask: Optional[torch.Tensor] = None  # (N, L)
        # -- constant tensors -- #
        self.register_buffer("idiom2subwords", idiom2subwords)  # (|W|, K)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 4, L);
         (num samples, 0=input_ids/1=token_type_ids/2=attention_mask/3=wisdom_mask, the maximum length)
        :return: (N, L, H); (num samples, k, the size of the vocabulary of subwords)
        """
        input_ids = X[:, 0]  # (N, 4, L) -> (N, L)
        token_type_ids = X[:, 1]  # (N, 4, L) -> (N, L)
        attention_mask = X[:, 2]  # (N, 4, L) -> (N, L)
        self.wisdom_mask = X[:, 3]  # (N, 4, L) -> (N, L)
        self.desc_mask = X[:, 4]  # (N, 4, L) -> (N, L)
        H_all = self.mlm.bert.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, H)
        return H_all

    def H_k(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        You may want to override this. (e.g. RDGamma - the k's could be anywhere)
        :param H_all (N, L, H)
        :return H_k (N, K, H)
        """
        N, _, H = H_all.size()
        # refer to: wisdomify/examples/explore_masked_select.py
        wisdom_mask = self.wisdom_mask.unsqueeze(2).expand(H_all.shape)  # (N, L) -> (N, L, 1) -> (N, L, H)
        H_k = torch.masked_select(H_all, wisdom_mask.bool())  # (N, L, H), (N, L, H) -> (N * K * H)
        H_k = H_k.reshape(N, self.hparams['k'], H)  # (N * K * H) -> (N, K, H)
        return H_k

    def H_desc(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        :param H_all (N, L, H)
        :return H_desc (N, L - (K + 3), H)
        """
        N, L, H = H_all.size()
        desc_mask = self.desc_mask.unsqueeze(2).expand(H_all.shape)
        H_desc = torch.masked_select(H_all, desc_mask.bool())  # (N, L, H), (N, L, H) -> (N * (L - (K + 3)) * H)
        H_desc = H_desc.reshape(N, L - (self.hparams['k'] + 3), H)  # (N * (L - (K + 3)) * H) -> (N, L - (K + 3), H)
        return H_desc

    def S_wisdom_literal(self, H_k: torch.Tensor) -> torch.Tensor:
        """
        To be used for both RDAlpha & RDBeta
        :param H_k: (N, K, H)
        :return: S_wisdom_literal (N, |W|)
        """
        S_vocab = self.mlm.cls(H_k)  # bmm; (N, K, H) * (H, |V|) ->  (N, K, |V|)
        indices = self.idiom2subwords.T.repeat(S_vocab.shape[0], 1, 1)  # (|W|, K) -> (N, K, |W|)
        S_wisdom_literal = S_vocab.gather(dim=-1, index=indices)  # (N, K, |V|) -> (N, K, |W|)
        S_wisdom_literal = S_wisdom_literal.sum(dim=1)  # (N, K, |W|) -> (N, |W|)
        return S_wisdom_literal

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        :param H_all: (N, L, H)
        :return S_wisdom: (N, |W|)
        """
        raise NotImplementedError("An RD class must implement S_wisdom")

    def P_wisdom(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 3, L)
        :return P_wisdom: (N, |W|), normalized over dim 1.
        """
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom = self.S_wisdom(H_all)  # (N, L, H) -> (N, W)
        P_wisdom = F.softmax(S_wisdom, dim=1)  # (N, W) -> (N, W)
        return P_wisdom

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        X, y = batch
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom = self.S_wisdom(H_all)  # (N, L, H) -> (N, |W|)
        loss = F.cross_entropy(S_wisdom, y)  # (N, |W|), (N,) -> (N,)
        loss = loss.sum()  # (N,) -> (1,)
        # so that the metrics accumulate over the course of this epoch
        # why dict? - just a boilerplate
        return {
            # you cannot change the keyword for the loss
            "loss": loss,
        }

    def on_train_batch_end(self, outputs: dict, *args, **kwargs) -> None:
        # watch the loss for this batch
        self.log("Train/Loss", outputs['loss'])

    def training_epoch_end(self, outputs: List[dict]) -> None:
        # to see an average performance over the batches in this specific epoch
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("Train/Average Loss", avg_loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        return self.training_step(batch, batch_idx)

    def on_validation_batch_end(self, outputs: dict, *args, **kwargs) -> None:
        self.log("Validation/Loss", outputs['loss'])

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        # to see an average performance over the batches in this specific epoch
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("Validation/Average Loss", avg_loss)

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


class Alpha(RD):
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


class BiLSTMPooler(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, batch_first=True,
                                  num_layers=1, bidirectional=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        hiddens, _ = self.lstm(X)
        return hiddens[:, -1]


class Gamma(RD):
    """
    @eubinecto
    S_wisdom  = S_wisdom_literal + S_wisdom_figurative
    but the way we get S_wisdom_figurative is much simplified, compared with RDBeta.
    """

    def __init__(self, mlm: BertForMaskedLM, idiom2subwords: torch.Tensor, k: int, lr: float):
        super().__init__(mlm, idiom2subwords, k, lr)
        # a pooler is a multilayer perceptron that pools wisdom_embeddings from idiom2subwords_embeddings
        self.pooler = BiLSTMPooler(self.mlm.config.hidden_size)
        # --- to be used to compute  attentions --- #
        self.attention_mask: Optional[torch.Tensor] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 4, L);
         (num samples, 0=input_ids/1=token_type_ids/2=attention_mask/3=wisdom_mask, the maximum length)
        :return: (N, L, H); (num samples, k, the size of the vocabulary of subwords)
        """
        input_ids = X[:, 0]  # (N, 4, L) -> (N, L)
        token_type_ids = X[:, 1]  # (N, 4, L) -> (N, L)
        self.attention_mask = X[:, 2]  # (N, 4, L) -> (N, L)
        self.wisdom_mask = X[:, 3]  # (N, 4, L) -> (N, L)
        self.desc_mask = X[:, 4]  # (N, 4, L) -> (N, L)
        H_all = self.mlm.bert.forward(input_ids, self.attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, H)
        return H_all

    def H_desc_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        this is needed mask the padding tokens
        :param attention_mask: (N, L)
        """
        N, L = attention_mask.size()
        H_desc_attention_mask = torch.masked_select(attention_mask, self.desc_mask.bool())
        H_desc_attention_mask = H_desc_attention_mask.reshape(N, L - (self.hparams['k'] + 3))
        return H_desc_attention_mask

    def S_wisdom(self, H_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S_wisdom_literal = self.S_wisdom_literal(self.H_k(H_all))
        S_wisdom_figurative = self.S_wisdom_figurative(H_all)
        S_wisdom = S_wisdom_literal + S_wisdom_figurative
        return S_wisdom, S_wisdom_literal, S_wisdom_figurative

    def S_wisdom_figurative(self, H_all: torch.Tensor) -> torch.Tensor:
        # --- draw the embeddings for wisdoms from  the embeddings of idiom2subwords -- #
        # this is to use as less of newly initialised weights as possible
        idiom2subwords_embeddings = self.mlm.bert \
            .embeddings.word_embeddings(self.idiom2subwords)  # (W, K)  -> (W, K, H)
        wisdom_embeddings = self.pooler(idiom2subwords_embeddings).squeeze()  # (W, H, K) -> (W, H, 1) -> (W, H)
        # --- draw H_wisdom from H_desc with attention --- #
        H_cls = H_all[:, 0]  # (N, L, H) -> (N, H)
        H_desc = self.H_desc(H_all)  # (N, L, H) -> (N, D, H)
        H_desc_attention_mask = self.H_desc_attention_mask(self.attention_mask)  # (N, L) -> (N, D)
        scores = torch.einsum("...h,...dh->...d", H_cls, H_desc)  # (N, D)
        # ignore the padding tokens
        scores = torch.masked_fill(scores, H_desc_attention_mask != 1, float("-inf"))  # (N, D)
        attentions = torch.softmax(scores, dim=1)  # over D
        H_wisdom = torch.einsum("...d,...dh->...h", attentions, H_desc)  # -> (N, H)
        # --- now compare H_wisdom with all the wisdoms --- #
        S_wisdom_figurative = torch.einsum("...h,wh->...w", H_wisdom, wisdom_embeddings)  # (N, H) * (W, H) -> (N, W)
        return S_wisdom_figurative

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        X, y = batch
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom, S_wisdom_literal, S_wisdom_figurative = self.S_wisdom(H_all)  # (N, L, H) -> (N, |W|)
        loss_all = F.cross_entropy(S_wisdom, y).sum()  # (N, |W|), (N,) -> (N,) -> (1,)
        loss_literal = F.cross_entropy(S_wisdom_literal, y).sum()  # (N, |W|), (N,) -> (N,) -> (1,)
        loss_figurative = F.cross_entropy(S_wisdom_figurative, y).sum()  # (N, |W|), (N,) -> (N,) -> (1,)
        loss = loss_all + loss_literal + loss_figurative  # unweighted multi-task learning
        return {
            # you cannot change the keyword for the loss
            "loss": loss,
        }

    def P_wisdom(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 3, L)
        :return P_wisdom: (N, |W|), normalized over dim 1.
        """
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom, _, _ = self.S_wisdom(H_all)  # (N, L, H) -> (N, W)
        P_wisdom = F.softmax(S_wisdom, dim=1)  # (N, W) -> (N, W)
        return P_wisdom
