from typing import Tuple, List
import yaml
import wandb
import pandas as pd
from idiomify.models import Alpha, Gamma
from idiomify.paths import idiom2def_dir, CONFIG_YAML, idioms_dir


# dataset
def fetch_idiom2def(ver: str) -> List[Tuple[str, str]]:
    artifact = wandb.Api().artifact(f"eubinecto/idiomify-demo/idiom2def:{ver}", type="dataset")
    artifact_path = idiom2def_dir(ver)
    artifact.download(root=str(artifact_path))
    tsv_path = artifact_path / "all.tsv"
    df = pd.read_csv(str(tsv_path), delimiter="\t")
    return [
        (row[0], row[1])
        for _, row in df.iterrows()
    ]


def fetch_idioms(ver: str) -> List[str]:
    artifact = wandb.Api().artifact(f"eubinecto/idiomify-demo/idioms:{ver}", type="dataset")
    artifact_path = idioms_dir(ver)
    artifact.download(root=str(artifact_path))
    tsv_path = artifact_path / "all.tsv"
    df = pd.read_csv(str(tsv_path), delimiter="\t")
    return [
        row[0]
        for _, row in df.iterrows()
    ]


# models
def fetch_alpha(ver: str) -> Alpha:
    pass


def fetch_gamma(ver: str) -> Gamma:
    pass


def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)
