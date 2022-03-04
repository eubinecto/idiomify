import torch
from typing import Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from idiomify.fetchers import fetch_idiom2def, fetch_epie
from idiomify.builders import Idiom2DefBuilder, Idiom2ContextBuilder, LabelsBuilder
from transformers import BertTokenizer


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


class Idiom2DefDataModule(LightningDataModule):

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
        X = Idiom2DefBuilder(self.tokenizer)(self.idiom2def, self.config['k'])
        y = LabelsBuilder(self.tokenizer)(self.idiom2def, self.idioms)
        self.dataset = IdiomifyDataset(X, y)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'])


class Idiom2ContextsDataModule(LightningDataModule):

    # boilerplate - just ignore these
    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def __init__(self, config: dict, tokenizer: BertTokenizer, idioms: List[str]):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.idioms = idioms
        self.idiom2context: Optional[List[Tuple[str, str]]] = None
        self.dataset: Optional[IdiomifyDataset] = None

    def prepare_data(self):
        """
        prepare: download all data needed for this from wandb to local.
        """
        self.idiom2context = [
            (idiom, context)
            for idiom, _, context in fetch_epie()
        ]

    def setup(self, stage: Optional[str] = None):
        # build the datasets
        X = Idiom2ContextBuilder(self.tokenizer)(self.idiom2context)
        y = LabelsBuilder(self.tokenizer)(self.idiom2context, self.idioms)
        self.dataset = IdiomifyDataset(X, y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'])
