import csv
from os import path
import yaml
import wandb
import requests
from typing import Tuple, List
from wandb.sdk.wandb_run import Run
from transformers import AutoModelForMaskedLM, AutoConfig, BertTokenizer
from idiomify.builders import Idiom2SubwordsBuilder
from idiomify.models import Alpha, RD
from idiomify.paths import CONFIG_YAML, idioms_dir, alpha_dir, literal2idiom
from idiomify.urls import (
    EPIE_IMMUTABLE_IDIOMS_URL,
    EPIE_IMMUTABLE_IDIOMS_CONTEXTS_URL,
    EPIE_IMMUTABLE_IDIOMS_TAGS_URL,
    EPIE_MUTABLE_IDIOMS_URL,
    EPIE_MUTABLE_IDIOMS_CONTEXTS_URL,
    EPIE_MUTABLE_IDIOMS_TAGS_URL,
    PIE_URL
)


# sources for dataset
def fetch_epie(ver: str) -> List[Tuple[str, str, str]]:
    """
    It fetches the EPIE idioms, contexts, and tags from the web
    :param ver: str
    :type ver: str
    :return: A list of tuples. Each tuple contains three strings: an idiom, a context, and a tag.
    """
    if ver == "immutable":
        idioms_url = EPIE_IMMUTABLE_IDIOMS_URL
        contexts_url = EPIE_IMMUTABLE_IDIOMS_CONTEXTS_URL
        tags_url = EPIE_IMMUTABLE_IDIOMS_TAGS_URL
    elif ver == "mutable":
        idioms_url = EPIE_MUTABLE_IDIOMS_URL
        contexts_url = EPIE_MUTABLE_IDIOMS_CONTEXTS_URL
        tags_url = EPIE_MUTABLE_IDIOMS_TAGS_URL
    else:
        raise ValueError
    idioms = requests.get(idioms_url).text
    contexts = requests.get(contexts_url).text
    tags = requests.get(tags_url).text
    return list(zip(idioms.strip().split("\n"),
                    contexts.strip().split("\n"),
                    tags.strip().split("\n")))


def fetch_pie() -> list:
    text = requests.get(PIE_URL).text
    lines = (line for line in text.split("\n") if line)
    reader = csv.reader(lines)
    next(reader)  # skip the header
    return [
        row
        for row in reader
    ]


# --- from wandb --- #
def fetch_idioms(ver: str, run: Run = None) -> List[str]:
    """
    why do you need this? -> you need this to have access to the idiom embeddings.
    """
    # if run object is given, we track the lineage of the data.
    # if not, we get the dataset via wandb Api.
    if run:
        artifact = run.use_artifact("idioms", type="dataset", aliases=ver)
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/idioms:{ver}", type="dataset")
    artifact_dir = artifact.download(root=idioms_dir(ver))
    txt_path = path.join(artifact_dir, "all.txt")
    with open(txt_path, 'r') as fh:
        return [line.strip() for line in fh]


def fetch_literal2idiom(ver: str, run: Run = None) -> List[Tuple[str, str]]:
    # if run object is given, we track the lineage of the data.
    # if not, we get the dataset via wandb Api.
    if run:
        artifact = run.use_artifact("literal2idiom", type="dataset", aliases=ver)
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/literal2idiom:{ver}", type="dataset")
    artifact_dir = artifact.download(root=literal2idiom(ver))
    tsv_path = path.join(artifact_dir, "all.tsv")
    with open(tsv_path, 'r') as fh:
        reader = csv.reader(fh, delimiter="\t")
        return [(row[0], row[1]) for row in reader]


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
    # if model == Alpha.name():
    #     rd = Alpha.load_from_checkpoint(str(ckpt_path), mlm=mlm, idiom2subwords=idiom2subwords)
    # elif model == Gamma.name():
    #     rd = Gamma.load_from_checkpoint(str(ckpt_path), mlm=mlm, idiom2subwords=idiom2subwords)
    # else:
    #     raise ValueError
    rd = ...
    return rd


def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)