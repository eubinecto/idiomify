import csv
import yaml
import wandb
from typing import Tuple, List
from idiomify.models import Alpha, Gamma, RD
from idiomify.paths import idiom2def_dir, CONFIG_YAML, idioms_dir, alpha_dir
from idiomify import tensors as T
from transformers import AutoModelForMaskedLM, AutoConfig, BertTokenizer


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
    idiom2subwords = T.idiom2subwords(idioms, tokenizer, config['k'])
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
