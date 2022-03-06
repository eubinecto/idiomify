import torch
import argparse
import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import BartTokenizer
from idiomify.datamodules import IdiomifyDataModule
from idiomify.fetchers import fetch_config, fetch_idiomifier
from idiomify.paths import ROOT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    args = parser.parse_args()
    config = fetch_config()['idiomifier']
    config.update(vars(args))
    tokenizer = BartTokenizer.from_pretrained(config['bart'])
    # prepare the datamodule
    with wandb.init(entity="eubinecto", project="idiomify", config=config) as run:
        model = fetch_idiomifier(config['ver'], run)  # fetch a pre-trained model
        datamodule = IdiomifyDataModule(config, tokenizer, run)
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(fast_dev_run=config['fast_dev_run'],
                             gpus=torch.cuda.device_count(),
                             default_root_dir=str(ROOT_DIR),
                             logger=logger)
        trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
