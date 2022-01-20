import wandb
import pandas as pd
from transformers import BertTokenizer
from idiomify.models import Alpha, Gamma
from idiomify.paths import wisdom2def_dir


# dataset
def fetch_wisdom2def(ver: str) -> pd.DataFrame:
    artifact = wandb.Api().artifact(f"eubinecto/idiomify-demo/wisdom2def:{ver}", type="dataset")
    artifact_path = wisdom2def_dir(ver)
    artifact.download(root=str(artifact_path))
    tsv_path = artifact_path / "all.tsv"
    df = pd.read_csv(str(tsv_path), delimiter="\t")
    return df


# models
def fetch_alpha(ver: str) -> Alpha:
    pass


def fetch_gamma(ver: str) -> Gamma:
    pass


def fetch_config() -> dict:
    pass


