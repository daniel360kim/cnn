#!/usr/bin/env python3
import argparse
import json
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


def load_ckpt(path):
    path = Path(path)
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=True)
    except TypeError:
        return torch.load(path, **kw)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def run_epoch(model, loader, crit, opt, device, train, desc=None):
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if desc else loader
    with ctx:
        for imgs, labels in it:
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
            if desc:
                it.set_postfix(loss="{:.4f}".format(loss.item()), acc="{:.4f}".format(correct / n))
    return loss_sum / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=cfg.ROOT / "config.toml")
    args = ap.parse_args()

    c = cfg.load(args.config)
    f = c["finetune"]
    paths = c["paths"]
    data_root = Path(paths["data_root"])
    ckpt_dir = Path(paths["ckpt_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = cfg.ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = torch.cuda.device_count() == 1
        print("device", device, torch.cuda.get_device_name(0))
    else:
        print("device", device, "(no cuda — check driver and CUDA_VISIBLE_DEVICES)")

    cls2idx, idx2cls = build_label_encoder(data_root / "train.csv")
    with open(ckpt_dir / "label_encoder.json", "w") as fp:
        json.dump(idx2cls, fp)

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
    w = int(f["workers"])
    bs = int(f["batch_size"])
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=w, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=w, pin_memory=True)
    print("train", len(tr_ds), "val", len(va_ds), "classes", len(idx2cls))

    backbone = ResNet18()
    ckpt = (f.get("backbone_ckpt") or "").strip()
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = cfg.ROOT / ckpt_path
        backbone.load_state_dict(load_ckpt(ckpt_path), strict=True)
        print("loaded backbone", ckpt_path)
    else:
        print("backbone from scratch")

    model = Classifier(backbone, num_classes=len(idx2cls))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    crit = nn.CrossEntropyLoss()

    probe_epochs = int(f.get("probe_epochs", 0))
    if ckpt and probe_epochs > 0:
        print("linear probe", probe_epochs, "epochs")
        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(False)
        opt_a = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9
        )
        sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=probe_epochs)
        for ep in range(1, probe_epochs + 1):
            _, tr_acc = run_epoch(
                model, tr_ld, crit, opt_a, device, True, desc="probe train {}/{}".format(ep, probe_epochs)
            )
            _, va_acc = run_epoch(
                model, va_ld, crit, None, device, False, desc="probe val {}/{}".format(ep, probe_epochs)
            )
            sch_a.step()
            print("probe", ep, "/", probe_epochs, "train acc", round(tr_acc, 4), "val acc", round(va_acc, 4))
        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(True)

    epochs = int(f["epochs"])
    print("finetune up to", epochs, "epochs, patience", int(f["patience"]))
    opt_b = torch.optim.Adam(
        [
            {"params": unwrap(model).backbone.parameters(), "lr": float(f["backbone_lr"])},
            {"params": unwrap(model).fc.parameters(), "lr": float(f["head_lr"])},
        ],
        weight_decay=1e-4,
    )
    sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=epochs)

    best, bad = 0.0, 0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, tr_ld, crit, opt_b, device, True, desc="train {}/{}".format(ep, epochs)
        )
        va_loss, va_acc = run_epoch(
            model, va_ld, crit, None, device, False, desc="val {}/{}".format(ep, epochs)
        )
        sch_b.step()
        lr0 = opt_b.param_groups[0]["lr"]
        print(
            "epoch",
            ep,
            "train",
            round(tr_acc, 4),
            round(tr_loss, 4),
            "val",
            round(va_acc, 4),
            round(va_loss, 4),
            "lr",
            lr0,
        )
        if va_acc > best:
            best = va_acc
            bad = 0
            torch.save(unwrap(model).state_dict(), ckpt_dir / "classifier_best.pth")
            print("  best val", round(best, 4))
        else:
            bad += 1
            if bad >= int(f["patience"]):
                print("early stop", ep)
                break

    print("best val", round(best, 4), ckpt_dir / "classifier_best.pth")


if __name__ == "__main__":
    main()
