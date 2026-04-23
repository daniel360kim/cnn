#!/usr/bin/env python3
"""Rank SimCLR backbone checkpoints by short linear-probe val accuracy (same split as finetune.py)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import gpu_env

gpu_env.set_visible_gpus()

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import cfg
from dataset import ButterflyDataset, build_label_encoder, get_supervised_transform
from models import Classifier, ResNet18


def load_sd(path: Path):
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=True)
    except TypeError:
        return torch.load(path, **kw)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def run_epoch(model, loader, crit, opt, device, train):
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = crit(logits, labels)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(imgs)
    return loss_sum / n, correct / n


def discover_backbones(ckpt_dir: Path) -> list[tuple[str, int | None, Path]]:
    """Return list of (label, sort_key, path). sort_key None for final -> sort last."""
    rows: list[tuple[str, int | None, Path]] = []
    for p in sorted(ckpt_dir.glob("simclr_epoch*.pth")):
        m = re.search(r"simclr_epoch(\d+)\.pth$", p.name)
        ep = int(m.group(1)) if m else 0
        rows.append(("epoch {:04d}".format(ep), ep, p))
    final = ckpt_dir / "simclr_final.pth"
    if final.is_file():
        rows.append(("final", None, final))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=cfg.ROOT / "config.toml")
    ap.add_argument(
        "--probe_epochs",
        type=int,
        default=None,
        help="Linear head epochs per checkpoint (default: finetune.probe_epochs or 5)",
    )
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--ckpt_dir",
        type=Path,
        default=None,
        help="Override [paths].ckpt_dir (default: from config)",
    )
    args = ap.parse_args()

    c = cfg.load(args.config)
    f = c["finetune"]
    paths = c["paths"]
    data_root = Path(paths["data_root"])
    ckpt_dir = args.ckpt_dir or Path(paths["ckpt_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = cfg.ROOT / ckpt_dir

    probe_epochs = int(args.probe_epochs if args.probe_epochs is not None else f.get("probe_epochs") or 5)
    bs = int(args.batch_size if args.batch_size is not None else f["batch_size"])
    w = int(args.workers if args.workers is not None else f["workers"])

    discovered = discover_backbones(ckpt_dir)
    if not discovered:
        raise SystemExit("no simclr_epoch*.pth or simclr_final.pth under {}".format(ckpt_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = torch.cuda.device_count() == 1

    cls2idx, idx2cls = build_label_encoder(data_root / "train.csv")
    df = pd.read_csv(data_root / "train.csv")
    tr, va = train_test_split(
        df, test_size=float(f["val_split"]), stratify=df["TARGET"], random_state=42
    )
    img_dir = data_root / "train_images" / "train_images"
    tr_ds = ButterflyDataset(
        tr.reset_index(drop=True), img_dir, get_supervised_transform(train=True), cls2idx
    )
    va_ds = ButterflyDataset(
        va.reset_index(drop=True), img_dir, get_supervised_transform(train=False), cls2idx
    )
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=w, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=w, pin_memory=True)

    crit = nn.CrossEntropyLoss()
    results: list[tuple[str, int | None, Path, float, float]] = []

    for label, sort_key, ckpt_path in tqdm(discovered, desc="checkpoints"):
        backbone = ResNet18()
        backbone.load_state_dict(load_sd(ckpt_path), strict=True)
        model = Classifier(backbone, num_classes=len(idx2cls))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(False)
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs)

        for _ in range(probe_epochs):
            run_epoch(model, tr_ld, crit, opt, device, True)
            sch.step()

        _, tr_acc = run_epoch(model, tr_ld, crit, None, device, False)
        _, va_acc = run_epoch(model, va_ld, crit, None, device, False)
        results.append((label, sort_key, ckpt_path, tr_acc, va_acc))

    results_by_val = sorted(results, key=lambda r: r[4], reverse=True)

    print("\n=== ranked by linear-probe val acc (probe_epochs={}) ===".format(probe_epochs))
    print("{:>12}  {:>8}  {:>8}  {}".format("label", "train", "val", "path"))
    for label, _, path, tra, vaa in results_by_val:
        print("{:>12}  {:>8.4f}  {:>8.4f}  {}".format(label, tra, vaa, path))

    best = results_by_val[0]
    print("\nbest: {}  val={:.4f}".format(best[2], best[4]))
    best_p = best[2].resolve()
    try:
        rel = best_p.relative_to(cfg.ROOT.resolve())
    except ValueError:
        rel = best_p
    print('set in config.toml:  backbone_ckpt = "{}"'.format(rel))


if __name__ == "__main__":
    main()
