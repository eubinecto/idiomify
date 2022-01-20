from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def wisdom2def_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"wisdom2def_{ver}"


def alpha_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"alpha_{ver}"


def gamma_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"beta_{ver}"
