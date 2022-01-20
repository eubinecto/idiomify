import torch
from typing import Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizer
from idiomify.fetchers import fetch_idiom2def
from idiomify import tensors as T


class IdiomifyDataset(Dataset):
    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]


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
                 tokenizer: BertTokenizer,
                 idioms: List[str]):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.idioms = idioms
        # --- to be downloaded & built --- #
        self.idiom2def: Optional[List[Tuple[str, str]]] = None
        self.dataset: Optional[IdiomifyDataset] = None

    def prepare_data(self):
        """
        prepare: download all data needed for this from wandb to local.
        """
        self.idiom2def = fetch_idiom2def(self.config['idiom2def_ver'])

    def setup(self, stage: Optional[str] = None):
        """
        setup the builders.
        """
        # --- set up the builders --- #
        # build the datasets
        X = T.inputs([definition for _, definition in self.idiom2def], self.tokenizer, self.config['k'])
        y = T.targets(self.idioms)
        self.dataset = IdiomifyDataset(X, y)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'])
