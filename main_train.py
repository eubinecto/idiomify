import os
import torch.cuda
import wandb
import argparse
import pytorch_lightning as pl
from termcolor import colored
from pytorch_lightning.loggers import WandbLogger
from transformers import BartForConditionalGeneration
from idiomify.datamodules import IdiomifyDataModule
from idiomify.fetchers import fetch_config, fetch_tokenizer
from idiomify.models import Idiomifier
from idiomify.paths import ROOT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--upload", dest='upload', action='store_true', default=False)
    args = parser.parse_args()
    config = fetch_config()['idiomifier']
    config.update(vars(args))
    if not config['upload']:
        print(colored("WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB", color="red"))
    # prepare a pre-trained BART
    bart = BartForConditionalGeneration.from_pretrained(config['bart'])
    # prepare the datamodule
    with wandb.init(entity="eubinecto", project="idiomify", config=config) as run:
        tokenizer = fetch_tokenizer(config['tokenizer_ver'], run)
        bart.resize_token_embeddings(len(tokenizer))  # because new tokens are added, this process is necessary
        model = Idiomifier(bart, config['lr'], tokenizer.bos_token_id, tokenizer.pad_token_id)
        datamodule = IdiomifyDataModule(config, tokenizer, run)
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             fast_dev_run=config['fast_dev_run'],
                             log_every_n_steps=config['log_every_n_steps'],
                             gpus=torch.cuda.device_count(),
                             default_root_dir=str(ROOT_DIR),
                             enable_checkpointing=False,
                             logger=logger)
        # start training
        trainer.fit(model=model, datamodule=datamodule)
        # upload the model to wandb only if the training is properly done  #
        if not config['fast_dev_run'] and trainer.current_epoch == config['max_epochs'] - 1:
            ckpt_path = ROOT_DIR / "model.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            config['vocab_size'] = len(tokenizer)  # this will be needed to fetch a pretrained idiomifier later
            artifact = wandb.Artifact(name="idiomifier", type="model", metadata=config)
            artifact.add_file(str(ckpt_path))
            run.log_artifact(artifact, aliases=["latest", config['ver']])
            os.remove(str(ckpt_path))  # make sure you remove it after you are done with uploading it


if __name__ == '__main__':
    main()
