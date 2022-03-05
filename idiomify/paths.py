from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CONFIG_YAML = ROOT_DIR / "config.yaml"


def idioms_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"idioms-{ver}"


def literal2idiomatic(ver: str) -> Path:
    return ARTIFACTS_DIR / f"literal2idiomatic-{ver}"


def seq2seq_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"seq2seq-{ver}"
