#!/usr/bin/env python3
"""Run inference on the test set; write CSV or Parquet with columns image_id, label."""
import argparse
import json
from pathlib import Path

import gpu_env

gpu_env.set_visible_gpus()

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import cfg
from dataset import get_supervised_transform
from models import Classifier, ResNet18


def load_ckpt(path):
    path = Path(path)
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=True)
    except TypeError:
        return torch.load(path, **kw)


def _collect_test_image_ids(test_dir):
    """Sorted stems under test_dir (jpg/jpeg/png). No sample_submission required."""
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise SystemExit("test directory does not exist: {}".format(test_dir))
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.stem)
    return [p.stem for p in paths]


class OrderedTestDataset(Dataset):
    """Loads test images in a fixed order (sorted image_id / stem)."""

    def __init__(self, test_dir, image_ids, transform):
        self.test_dir = Path(test_dir)
        self.ids = list(image_ids)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.test_dir / (stem + ext)
            if p.is_file():
                path = p
                break
        if path is None:
            raise FileNotFoundError("no image for id {} under {}".format(stem, self.test_dir))
        img = Image.open(path).convert("RGB")
        return self.transform(img), stem


def _write_table(df, out_path, fmt):
    out_path = Path(out_path)
    fmt = (fmt or "auto").lower()
    if fmt == "auto":
        fmt = "parquet" if out_path.suffix.lower() == ".parquet" else "csv"
    if fmt == "parquet":
        try:
            df.to_parquet(out_path, index=False)
        except ImportError as ex:
            raise SystemExit(
                "Parquet output needs pyarrow (or fastparquet). Try: pip install pyarrow\n{}".format(ex)
            ) from ex
    elif fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        raise SystemExit("unknown --format {}; use csv, parquet, or auto".format(fmt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=cfg.ROOT / "config.toml")
    ap.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="classifier state_dict (.pth). Default: <ckpt_dir>/classifier_best.pth",
    )
    ap.add_argument(
        "--label-encoder",
        type=Path,
        default=None,
        help="label_encoder.json from finetune. Default: <ckpt_dir>/label_encoder.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path. Default: <ckpt_dir>/submission.csv (.parquet also supported)",
    )
    ap.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="folder of test images; row order = sorted stems. Implies ignoring sample_submission.csv.",
    )
    ap.add_argument(
        "--format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="output format (default: infer from --output extension)",
    )
    ap.add_argument(
        "--expect-rows",
        type=int,
        default=1000,
        help="if > 0, warn when row count differs (typical class test set size)",
    )
    ap.add_argument(
        "--strict-rows",
        action="store_true",
        help="exit with error if row count != --expect-rows (requires --expect-rows > 0)",
    )
    args = ap.parse_args()

    c = cfg.load(args.config)
    paths = c["paths"]
    data_root = Path(paths["data_root"])
    ckpt_dir = Path(paths["ckpt_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = cfg.ROOT / ckpt_dir

    weights = args.weights or (ckpt_dir / "classifier_best.pth")
    enc_path = args.label_encoder or (ckpt_dir / "label_encoder.json")
    out_path = args.output or (ckpt_dir / "submission.csv")

    if not weights.is_file():
        raise SystemExit("missing weights: {} (run finetune.py first)".format(weights))
    if not enc_path.is_file():
        raise SystemExit("missing label encoder: {} (run finetune.py first)".format(enc_path))

    with open(enc_path) as fp:
        idx2cls = json.load(fp)
    if not isinstance(idx2cls, list):
        raise SystemExit("label_encoder.json should be a JSON list of class names")

    default_test_dir = data_root / "test_images" / "test_images"
    test_dir = Path(args.test_dir) if args.test_dir is not None else default_test_dir
    sample = data_root / "sample_submission.csv"
    use_sample = args.test_dir is None and sample.is_file()
    if use_sample:
        template = pd.read_csv(sample)
        id_cols = [c for c in template.columns if str(c).strip().lstrip("\ufeff").lower() == "image_id"]
        if not id_cols:
            print(
                "warning: {} has no image_id column; using sorted images under {}".format(
                    sample, test_dir
                )
            )
            image_ids = _collect_test_image_ids(test_dir)
        else:
            image_ids = template[id_cols[0]].astype(str).tolist()
            print("row order from", sample, "n=", len(image_ids))
    else:
        if args.test_dir is not None:
            print("using --test-dir only (ignoring sample_submission.csv if present)")
        image_ids = _collect_test_image_ids(test_dir)
        print("row order: sorted stems under", test_dir, "n=", len(image_ids))

    if args.expect_rows > 0 and len(image_ids) != args.expect_rows:
        msg = "row count {} != --expect-rows {} (submission spec may require exactly {}).".format(
            len(image_ids), args.expect_rows, args.expect_rows
        )
        if args.strict_rows:
            raise SystemExit(msg)
        print("warning:", msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = torch.cuda.device_count() == 1

    f = c.get("finetune", {})
    bs = int(f.get("batch_size", 128))
    w = int(f.get("workers", 4))

    ds = OrderedTestDataset(test_dir, image_ids, get_supervised_transform(train=False))
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=w,
        pin_memory=device.type == "cuda",
    )

    backbone = ResNet18()
    model = Classifier(backbone, num_classes=len(idx2cls))
    model.load_state_dict(load_ckpt(weights), strict=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for imgs, _stems in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            pred = logits.argmax(1).cpu().tolist()
            preds.extend(idx2cls[i] for i in pred)

    if len(preds) != len(image_ids):
        raise SystemExit("internal error: pred count mismatch")

    out = pd.DataFrame({"image_id": image_ids, "label": preds})
    _write_table(out, out_path, args.format)
    print("wrote", out_path, "rows", len(out), "columns", list(out.columns))


if __name__ == "__main__":
    main()
