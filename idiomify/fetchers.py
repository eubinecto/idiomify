import yaml
import wandb
from os import path
import pandas as pd
from typing import Tuple, List
from wandb.sdk.wandb_run import Run
from idiomify.paths import CONFIG_YAML, idioms_dir, literal2idiomatic, seq2seq_dir
from idiomify.urls import PIE_URL
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from idiomify.models import Idiomifier


def fetch_pie() -> pd.DataFrame:
    # fetch & parse it directly from the web
    return pd.read_csv(PIE_URL)


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


def fetch_literal2idiomatic(ver: str, run: Run = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # if run object is given, we track the lineage of the data.
    # if not, we get the dataset via wandb Api.
    if run:
        artifact = run.use_artifact(f"literal2idiomatic:{ver}", type="dataset")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/literal2idiomatic:{ver}", type="dataset")
    artifact_dir = artifact.download(root=literal2idiomatic(ver))
    train_path = path.join(artifact_dir, "train.tsv")
    test_path = path.join(artifact_dir, "test.tsv")
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    return train_df, test_df


def fetch_idiomifier(ver: str, run: Run = None) -> Idiomifier:
    """
    you may want to change the name to Idiomifier.
    The current Idiomifier then turns into a pipeline.
    """
    if run:
        artifact = run.use_artifact(f"seq2seq:{ver}", type="model")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/seq2seq:{ver}", type="model")
    config = artifact.metadata
    artifact_dir = artifact.download(root=seq2seq_dir(ver))
    ckpt_path = path.join(artifact_dir, "model.ckpt")
    bart = AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(config['bart']))
    alpha = Idiomifier.load_from_checkpoint(ckpt_path, bart=bart)
    return alpha


def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)
