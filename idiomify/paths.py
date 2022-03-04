from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CONFIG_YAML = ROOT_DIR / "config.yaml"


def idioms_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"idioms_{ver}"


def literal2idiom(ver: str) -> Path:
    return ARTIFACTS_DIR / f"literal2idiom_{ver}"


def alpha_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"alpha_{ver}"


def gamma_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"beta_{ver}"
