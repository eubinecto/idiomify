import torch
from typing import Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from wandb.sdk.wandb_run import Run

from idiomify.fetchers import fetch_literal2idiomatic
from idiomify.builders import SourcesBuilder, TargetsBuilder, TargetsRightShiftedBuilder
from transformers import BartTokenizer


class IdiomifyDataset(Dataset):
    def __init__(self,
                 srcs: torch.Tensor,
                 tgts_r: torch.Tensor,
                 tgts: torch.Tensor):
        self.srcs = srcs  # (N, 2, L)
        self.tgts_r = tgts_r  # (N, 2, L)
        self.tgts = tgts  # (N, L)

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        assert self.srcs.shape[0] == self.tgts_r.shape[0] == self.tgts.shape[0]
        return self.srcs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.srcs[idx], self.tgts_r[idx], self.tgts[idx]


class IdiomifyDataModule(LightningDataModule):

    # boilerplate - just ignore these
    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def __init__(self,
                 config: dict,
                 tokenizer: BartTokenizer,
                 run: Run = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.run = run
        # --- to be downloaded & built --- #
        self.literal2idiomatic: Optional[List[Tuple[str, str]]] = None
        self.dataset: Optional[IdiomifyDataset] = None

    def prepare_data(self):
        """
        prepare: download all data needed for this from wandb to local.
        """
        self.literal2idiomatic = fetch_literal2idiomatic(self.config['literal2idiomatic_ver'], self.run)

    def setup(self, stage: Optional[str] = None):
        # --- set up the builders --- #
        # build the datasets
        srcs = SourcesBuilder(self.tokenizer)(self.literal2idiomatic)
        tgts_r = TargetsRightShiftedBuilder(self.tokenizer)(self.literal2idiomatic)
        tgts = TargetsBuilder(self.tokenizer)(self.literal2idiomatic)
        self.dataset = IdiomifyDataset(srcs, tgts_r, tgts)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'])
