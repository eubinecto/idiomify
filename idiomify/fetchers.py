import csv
from os import path
import yaml
import wandb
import requests
from typing import Tuple, List
from wandb.sdk.wandb_run import Run
from idiomify.paths import CONFIG_YAML, idioms_dir, literal2idiomatic, alpha_dir
from idiomify.urls import (
    EPIE_IMMUTABLE_IDIOMS_URL,
    EPIE_IMMUTABLE_IDIOMS_CONTEXTS_URL,
    EPIE_IMMUTABLE_IDIOMS_TAGS_URL,
    EPIE_MUTABLE_IDIOMS_URL,
    EPIE_MUTABLE_IDIOMS_CONTEXTS_URL,
    EPIE_MUTABLE_IDIOMS_TAGS_URL,
    PIE_URL
)
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from models import Alpha


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
        artifact = run.use_artifact(f"idioms:{ver}", type="dataset")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/idioms:{ver}", type="dataset")
    artifact_dir = artifact.download(root=idioms_dir(ver))
    txt_path = path.join(artifact_dir, "all.txt")
    with open(txt_path, 'r') as fh:
        return [line.strip() for line in fh]


def fetch_literal2idiomatic(ver: str, run: Run = None) -> List[Tuple[str, str]]:
    # if run object is given, we track the lineage of the data.
    # if not, we get the dataset via wandb Api.
    if run:
        artifact = run.use_artifact(f"literal2idiomatic:{ver}", type="dataset")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/literal2idiomatic:{ver}", type="dataset")
    artifact_dir = artifact.download(root=literal2idiomatic(ver))
    tsv_path = path.join(artifact_dir, "all.tsv")
    with open(tsv_path, 'r') as fh:
        reader = csv.reader(fh, delimiter="\t")
        return [(row[0], row[1]) for row in reader]


def fetch_alpha(ver: str, run: Run = None) -> Alpha:
    if run:
        artifact = run.use_artifact(f"alpha:{ver}", type="model")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/alpha:{ver}", type="model")
    config = artifact.metadata
    artifact_dir = artifact.download(root=alpha_dir(ver))
    ckpt_path = path.join(artifact_dir, "model.ckpt")
    bart = AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(config['bart']))
    alpha = Alpha.load_from_checkpoint(ckpt_path, bart=bart)
    return alpha


def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)
