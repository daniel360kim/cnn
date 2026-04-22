#!/usr/bin/env python3
import argparse
import shutil
import sys
import tomllib
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.toml")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        e = tomllib.load(f).get("etl", {})
    comp = e.get("competition", "elec-378-sp-26-final-project")
    out = Path(e.get("output_dir", "etl/raw"))
    if not out.is_absolute():
        out = ROOT / out

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    path = kagglehub.competition_download(
        comp, force_download=args.force_download, output_dir=str(out)
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
