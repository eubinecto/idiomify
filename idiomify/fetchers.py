import yaml
import wandb
from os import path
import pandas as pd
from typing import Tuple
from wandb.sdk.wandb_run import Run
from idiomify.paths import CONFIG_YAML, idioms_dir, literal2entities_dir, idiomifier_dir
from idiomify.urls import PIE_URL
from transformers import AutoModelForSeq2SeqLM, AutoConfig, BartTokenizer
from idiomify.models import Idiomifier
from idiomify.pipeline import Pipeline


# --- from the web --- #
def fetch_pie() -> pd.DataFrame:
    # fetch & parse it directly from the web
    return pd.read_csv(PIE_URL)


# --- from wandb --- #
def fetch_idioms(ver: str, run: Run = None) -> pd.DataFrame:
    """
    why do you need this? -> you need this to have access to the idiom embeddings.
    """
    # if run object is given, we track the lineage of the data.
    # if not, we get the dataset via wandb Api.
    if run:
        artifact = run.use_artifact(f"idioms:{ver}", type="dataset")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/idioms:{ver}", type="dataset")
    artifact_dir = artifact.download(root=str(idioms_dir(ver)))
    tsv_path = path.join(artifact_dir, "all.tsv")
    return pd.read_csv(tsv_path, sep="\t")


def fetch_literal2entities(ver: str, run: Run = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # if run object is given, we track the lineage of the data.
    # if not, we get the dataset via wandb Api.
    if run:
        artifact = run.use_artifact(f"literal2labels:{ver}", type="dataset")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/literal2entities:{ver}", type="dataset")
    artifact_dir = artifact.download(root=str(literal2entities_dir(ver)))
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
        artifact = run.use_artifact(f"idiomifier:{ver}", type="model")
    else:
        artifact = wandb.Api().artifact(f"eubinecto/idiomify/idiomifier:{ver}", type="model")
    config = artifact.metadata
    artifact_dir = artifact.download(root=str(idiomifier_dir(ver)))
    ckpt_path = path.join(artifact_dir, "model.ckpt")
    bart = AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(config['bart']))
    bart.resize_token_embeddings(config['vocab_size'])
    model = Idiomifier.load_from_checkpoint(ckpt_path, bart=bart)
    return model


def fetch_pipeline() -> Pipeline:
    """
    fetch a pipeline of the version stated in config.yaml
    """
    config = fetch_config()['idiomifier']
    model = fetch_idiomifier(config['ver'])
    tokenizer = ... # just get a pre-trained one.
    idioms = fetch_idioms(config['idioms_ver'])
    model.eval()  # this is crucial to obtain consistent results
    pipeline = Pipeline(model, tokenizer, idioms)
    return pipeline


# --- from local --- #
def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)
