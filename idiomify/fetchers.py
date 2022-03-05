import csv
from os import path
import yaml
import wandb
import requests
from typing import Tuple, List
from wandb.sdk.wandb_run import Run
from idiomify.paths import CONFIG_YAML, idioms_dir, literal2idiomatic, seq2seq_dir
from idiomify.urls import PIE_URL
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from idiomify.models import Seq2Seq


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


def fetch_seq2seq(ver: str, run: Run = None) -> Seq2Seq:
    if run:
        artifact = run.use_artifact(f"seq2seq:{ver}", type="model")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/seq2seq:{ver}", type="model")
    config = artifact.metadata
    artifact_dir = artifact.download(root=seq2seq_dir(ver))
    ckpt_path = path.join(artifact_dir, "model.ckpt")
    bart = AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(config['bart']))
    alpha = Seq2Seq.load_from_checkpoint(ckpt_path, bart=bart)
    return alpha


def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)
