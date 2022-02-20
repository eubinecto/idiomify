import csv
import yaml
import wandb
import requests
from typing import Tuple, List

from wandb.sdk.wandb_run import Run

from idiomify.models import Alpha, RD
from idiomify.paths import idiom2def_dir, CONFIG_YAML, idioms_dir, alpha_dir
from idiomify.urls import (
    EPIE_IMMUTABLE_IDIOMS_URL,
    EPIE_IMMUTABLE_IDIOMS_CONTEXTS_URL,
    EPIE_IMMUTABLE_IDIOMS_TAGS_URL,
    EPIE_MUTABLE_IDIOMS_URL,
    EPIE_MUTABLE_IDIOMS_CONTEXTS_URL,
    EPIE_MUTABLE_IDIOMS_TAGS_URL
)
from idiomify.builders import Idiom2SubwordsBuilder
from transformers import AutoModelForMaskedLM, AutoConfig, BertTokenizer


# sources for dataset
def fetch_epie() -> List[Tuple[str, str, str]]:
    idioms = requests.get(EPIE_IMMUTABLE_IDIOMS_URL).text \
             + requests.get(EPIE_MUTABLE_IDIOMS_URL).text
    contexts = requests.get(EPIE_IMMUTABLE_IDIOMS_CONTEXTS_URL).text \
               + requests.get(EPIE_MUTABLE_IDIOMS_CONTEXTS_URL).text
    tags = requests.get(EPIE_IMMUTABLE_IDIOMS_TAGS_URL).text \
           + requests.get(EPIE_MUTABLE_IDIOMS_TAGS_URL).text
    return list(zip(idioms.strip().split("\n"),
                    contexts.strip().split("\n"),
                    tags.strip().split("\n")))


# you should somehow get this from... wandb.
def fetch_idiom2context(ver: str, run: Run = None) -> List[Tuple[str, str]]:
    """
    include run if you want to track the lineage
    """
    if run:
        pass


# dataset
def fetch_idiom2def(ver: str) -> List[Tuple[str, str]]:
    artifact = wandb.Api().artifact(f"eubinecto/idiomify-demo/idiom2def:{ver}", type="dataset")
    artifact_path = idiom2def_dir(ver)
    artifact.download(root=str(artifact_path))
    tsv_path = artifact_path / "all.tsv"
    with open(tsv_path, 'r') as fh:
        reader = csv.reader(fh, delimiter="\t")
        return [
            (row[0], row[1])
            for row in reader
        ]


def fetch_idioms(ver: str) -> List[str]:
    artifact = wandb.Api().artifact(f"eubinecto/idiomify-demo/idioms:{ver}", type="dataset")
    artifact_path = idioms_dir(ver)
    artifact.download(root=str(artifact_path))
    tsv_path = artifact_path / "all.tsv"
    with open(tsv_path, 'r') as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        return [
            row[0]
            for row in reader
        ]


def fetch_rd(model: str, ver: str) -> RD:
    artifact = wandb.Api().artifact(f"eubinecto/idiomify-demo/{model}:{ver}", type="model")
    config = artifact.metadata
    artifact_path = alpha_dir(ver)
    artifact.download(root=str(artifact_path))
    mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(config['bert']))
    ckpt_path = artifact_path / "rd.ckpt"
    idioms = fetch_idioms(config['idioms_ver'])
    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    idiom2subwords = Idiom2SubwordsBuilder(tokenizer)(idioms, config['k'])
    if model == Alpha.name():
        rd = Alpha.load_from_checkpoint(str(ckpt_path), mlm=mlm, idiom2subwords=idiom2subwords)
    elif model == Gamma.name():
        rd = Gamma.load_from_checkpoint(str(ckpt_path), mlm=mlm, idiom2subwords=idiom2subwords)
    else:
        raise ValueError
    return rd


def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)
