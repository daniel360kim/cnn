import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path=None):
    p = Path(path or os.environ.get("CNN_CONFIG", ROOT / "config.toml"))
    with open(p, "rb") as f:
        return tomllib.load(f)
