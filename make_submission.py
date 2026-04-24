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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import cfg
from dataset import get_supervised_transform, _MEAN, _STD
from models import Classifier, build_backbone


def load_ckpt(path):
    path = Path(path)
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=True)
    except TypeError:
        return torch.load(path, **kw)


def _collect_test_image_ids(test_dir):
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise SystemExit("test directory does not exist: {}".format(test_dir))
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.stem)
    return [p.stem for p in paths]


def _open_image(test_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(test_dir) / (stem + ext)
        if p.is_file():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError("no image for id {} under {}".format(stem, test_dir))


def _build_tta_transforms(size):
    """Returns a list of transforms for TTA: original, H-flip, V-flip."""
    val_size = int(size * 256 / 224)
    base = [
        transforms.Resize(val_size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
    hflip = [
        transforms.Resize(val_size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
    vflip = [
        transforms.Resize(val_size),
        transforms.CenterCrop(size),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
    return [transforms.Compose(t) for t in [base, hflip, vflip]]


class OrderedTestDataset(Dataset):
    def __init__(self, test_dir, image_ids, transform):
        self.test_dir = Path(test_dir)
        self.ids = list(image_ids)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = _open_image(self.test_dir, self.ids[idx])
        return self.transform(img), self.ids[idx]


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
                "Parquet output needs pyarrow. Try: pip install pyarrow\n{}".format(ex)
            ) from ex
    elif fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        raise SystemExit("unknown --format {}; use csv, parquet, or auto".format(fmt))


def _load_model(weights_path, backbone_name, num_classes, dropout, device):
    backbone = build_backbone(backbone_name)
    model = Classifier(backbone, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(load_ckpt(weights_path), strict=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    return model


def _infer_single_transform(model, loader, num_classes, device):
    """Returns (N, C) softmax probabilities."""
    probs_list = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs_list.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs_list, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=cfg.ROOT / "config.toml")
    ap.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more classifier state_dict files (.pth). "
            "Defaults to all classifier_fold*.pth in ckpt_dir, "
            "falling back to classifier_best.pth."
        ),
    )
    ap.add_argument("--label-encoder", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--test-dir", type=Path, default=None)
    ap.add_argument("--format", choices=("auto", "csv", "parquet"), default="auto")
    ap.add_argument("--expect-rows", type=int, default=1000)
    ap.add_argument("--strict-rows", action="store_true")
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation (3 views)")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.set_defaults(tta=None)  # None = read from config
    args = ap.parse_args()

    c = cfg.load(args.config)
    paths = c["paths"]
    f = c.get("finetune", {})
    data_root = Path(paths["data_root"])
    ckpt_dir = Path(paths["ckpt_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = cfg.ROOT / ckpt_dir

    enc_path = args.label_encoder or (ckpt_dir / "label_encoder.json")
    out_path = args.output or (ckpt_dir / "submission.csv")

    if not enc_path.is_file():
        raise SystemExit("missing label encoder: {} (run finetune.py first)".format(enc_path))
    with open(enc_path) as fp:
        idx2cls = json.load(fp)
    num_classes = len(idx2cls)

    # Resolve checkpoint list
    if args.weights:
        weights_list = [Path(w) for w in args.weights]
    else:
        fold_ckpts = sorted(ckpt_dir.glob("classifier_fold*.pth"))
        weights_list = fold_ckpts if fold_ckpts else [ckpt_dir / "classifier_best.pth"]
    for w in weights_list:
        if not w.is_file():
            raise SystemExit("missing weights: {}".format(w))
    print("checkpoints ({}):\n  {}".format(len(weights_list), "\n  ".join(str(w) for w in weights_list)))

    # TTA flag: CLI > config
    use_tta = args.tta
    if use_tta is None:
        use_tta = bool(f.get("tta", False))
    print("TTA:", use_tta)

    default_test_dir = data_root / "test_images" / "test_images"
    test_dir = Path(args.test_dir) if args.test_dir is not None else default_test_dir
    sample = data_root / "sample_submission.csv"
    use_sample = args.test_dir is None and sample.is_file()
    if use_sample:
        template = pd.read_csv(sample)
        id_cols = [c for c in template.columns if str(c).strip().lstrip("﻿").lower() == "image_id"]
        if not id_cols:
            image_ids = _collect_test_image_ids(test_dir)
        else:
            image_ids = template[id_cols[0]].astype(str).tolist()
            print("row order from", sample, "n=", len(image_ids))
    else:
        image_ids = _collect_test_image_ids(test_dir)
        print("row order: sorted stems under", test_dir, "n=", len(image_ids))

    if args.expect_rows > 0 and len(image_ids) != args.expect_rows:
        msg = "row count {} != --expect-rows {}".format(len(image_ids), args.expect_rows)
        if args.strict_rows:
            raise SystemExit(msg)
        print("warning:", msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = torch.cuda.device_count() == 1

    img_size = int(f.get("img_size", 224))
    bs = int(f.get("batch_size", 128))
    w = int(f.get("workers", 4))
    backbone_name = f.get("backbone", "resnet18")
    dropout = float(f.get("dropout", 0.0))

    # Build transforms
    if use_tta:
        tta_transforms = _build_tta_transforms(img_size)
        print("TTA views:", len(tta_transforms))
    else:
        base_transform = get_supervised_transform(img_size, train=False)

    # Accumulate softmax averaged over all models × TTA views
    total_probs = torch.zeros(len(image_ids), num_classes)

    num_passes = len(weights_list) * (len(tta_transforms) if use_tta else 1)
    print("total forward passes per image:", num_passes)

    for wi, weights_path in enumerate(weights_list):
        print("loading model {}/{}: {}".format(wi + 1, len(weights_list), weights_path))
        model = _load_model(weights_path, backbone_name, num_classes, dropout, device)

        if use_tta:
            for ti, tfm in enumerate(tta_transforms):
                ds = OrderedTestDataset(test_dir, image_ids, tfm)
                loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=w,
                                    pin_memory=device.type == "cuda")
                probs = _infer_single_transform(model, loader, num_classes, device)
                total_probs += probs
                print("  tta view {}/{}".format(ti + 1, len(tta_transforms)))
        else:
            ds = OrderedTestDataset(test_dir, image_ids, base_transform)
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=w,
                                pin_memory=device.type == "cuda")
            probs = _infer_single_transform(model, loader, num_classes, device)
            total_probs += probs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preds = total_probs.argmax(dim=1).tolist()
    labels = [idx2cls[i] for i in preds]

    if len(labels) != len(image_ids):
        raise SystemExit("internal error: pred count mismatch")

    out = pd.DataFrame({"image_id": image_ids, "label": labels})
    _write_table(out, out_path, args.format)
    print("wrote", out_path, "rows", len(out))


if __name__ == "__main__":
    main()
