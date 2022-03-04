import os
import torch.cuda
import wandb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from termcolor import colored
from transformers import BertForMaskedLM, BertTokenizer
from idiomify.datamodules import IdiomifyDataModule
from idiomify.fetchers import fetch_config, fetch_idioms
from idiomify.models import Alpha, Gamma
from idiomify.paths import ROOT_DIR
from idiomify import tensors as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpha")
    parser.add_argument("--ver", type=str, default="eng2eng")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--upload", dest='upload', action='store_true', default=False)
    args = parser.parse_args()
    config = fetch_config()[args.model][args.ver]
    config.update(vars(args))
    if not config['upload']:
        print(colored("WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB", color="red"))

    # prepare arguments
    mlm = BertForMaskedLM.from_pretrained(config['bert'])
    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    idioms = fetch_idioms(config['idioms_ver'])
    idiom2subwords = T.idiom2subwords(idioms, tokenizer, config['k'])
    # choose the model to train
    if config['model'] == Alpha.name():
        rd = Alpha(mlm, idiom2subwords, config['k'], config['lr'])
    elif config['model'] == Gamma.name():
        rd = Gamma(mlm, idiom2subwords, config['k'], config['lr'])
    else:
        raise ValueError
    # prepare datamodule
    datamodule = IdiomifyDataModule(config, tokenizer, idioms)

    with wandb.init(entity="eubinecto", project="idiomify-demo", config=config) as run:
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             fast_dev_run=config['fast_dev_run'],
                             log_every_n_steps=config['log_every_n_steps'],
                             gpus=torch.cuda.device_count(),
                             default_root_dir=str(ROOT_DIR),
                             enable_checkpointing=False,
                             logger=logger)
        # start training
        trainer.fit(model=rd, datamodule=datamodule)
        # upload the model to wandb only if the training is properly done  #
        if not config['fast_dev_run'] and trainer.current_epoch == config['max_epochs'] - 1:
            ckpt_path = ROOT_DIR / "rd.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            artifact = wandb.Artifact(name=config['model'], type="model", metadata=config)
            artifact.add_file(str(ckpt_path))
            run.log_artifact(artifact, aliases=["latest", config['ver']])
            os.remove(str(ckpt_path))  # make sure you remove it after you are done with uploading it


if __name__ == '__main__':
    main()
