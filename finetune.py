#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path

import gpu_env

gpu_env.set_visible_gpus()

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import cfg
from dataset import ButterflyDataset, build_label_encoder, get_supervised_transform
from models import Classifier, build_backbone


def load_ckpt(path):
    path = Path(path)
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=True)
    except TypeError:
        return torch.load(path, **kw)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def mixup_batch(imgs, labels, alpha, device):
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    n = imgs.size(0)
    idx = torch.randperm(n, device=device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


def cutmix_batch(imgs, labels, alpha, device):
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    n, _, H, W = imgs.shape
    idx = torch.randperm(n, device=device)
    cut_w = int(W * math.sqrt(1 - lam))
    cut_h = int(H * math.sqrt(1 - lam))
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[idx], lam


def mixup_loss(crit, logits, y_a, y_b, lam):
    return lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)


def make_lr_lambda(warmup_epochs, total_epochs):
    def fn(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        p = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return fn


def run_epoch(model, loader, crit, opt, device, train, mixup_alpha=0.0, cutmix_alpha=0.0, desc=None):
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if desc else loader
    with ctx:
        for imgs, labels in it:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train and (mixup_alpha > 0 or cutmix_alpha > 0):
                use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(1).item() > 0.5)
                if use_cutmix:
                    imgs, y_a, y_b, lam = cutmix_batch(imgs, labels, cutmix_alpha, device)
                else:
                    imgs, y_a, y_b, lam = mixup_batch(imgs, labels, mixup_alpha, device)
                logits = model(imgs)
                loss = mixup_loss(crit, logits, y_a, y_b, lam)
            else:
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


def build_model(f, num_classes, device):
    backbone_name = f.get("backbone", "resnet18")
    backbone = build_backbone(backbone_name)
    ckpt = (f.get("backbone_ckpt") or "").strip()
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = cfg.ROOT / ckpt_path
        backbone.load_state_dict(load_ckpt(ckpt_path), strict=True)
        print("loaded backbone", ckpt_path)
    else:
        print("backbone from scratch ({})".format(backbone_name))
    dropout = float(f.get("dropout", 0.0))
    model = Classifier(backbone, num_classes=num_classes, dropout=dropout)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(device)


def train_fold(f, tr, va, img_dir, cls2idx, idx2cls, device, ckpt_path, fold_label):
    img_size = int(f.get("img_size", 224))
    bs = int(f["batch_size"])
    w = int(f["workers"])
    aug_strength = f.get("aug", "default")
    tr_ds = ButterflyDataset(tr.reset_index(drop=True), img_dir, get_supervised_transform(img_size, train=True, strength=aug_strength), cls2idx)
    va_ds = ButterflyDataset(va.reset_index(drop=True), img_dir, get_supervised_transform(img_size, train=False), cls2idx)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=w, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=w, pin_memory=True)
    print("{} train {} val {}".format(fold_label, len(tr_ds), len(va_ds)))

    model = build_model(f, len(idx2cls), device)
    crit = nn.CrossEntropyLoss(label_smoothing=float(f.get("label_smoothing", 0.0)))
    mixup_alpha = float(f.get("mixup_alpha", 0.0))
    cutmix_alpha = float(f.get("cutmix_alpha", 0.0))

    probe_epochs = int(f.get("probe_epochs", 0))
    ckpt = (f.get("backbone_ckpt") or "").strip()
    if ckpt and probe_epochs > 0:
        print("linear probe", probe_epochs, "epochs")
        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(False)
        opt_a = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9
        )
        sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=probe_epochs)
        for ep in range(1, probe_epochs + 1):
            _, tr_acc = run_epoch(model, tr_ld, crit, opt_a, device, True,
                                  desc="probe train {}/{}".format(ep, probe_epochs))
            _, va_acc = run_epoch(model, va_ld, crit, None, device, False,
                                  desc="probe val {}/{}".format(ep, probe_epochs))
            sch_a.step()
            print("probe", ep, "/", probe_epochs, "train acc", round(tr_acc, 4), "val acc", round(va_acc, 4))
        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(True)

    epochs = int(f["epochs"])
    warmup_epochs = int(f.get("warmup_epochs", 0))
    patience = int(f["patience"])
    print("finetune {} epochs, warmup {}, patience {}".format(epochs, warmup_epochs, patience))

    wd = float(f.get("weight_decay", 1e-4))
    opt_b = torch.optim.AdamW(
        [
            {"params": unwrap(model).backbone.parameters(), "lr": float(f["backbone_lr"])},
            {"params": unwrap(model).fc.parameters(), "lr": float(f["head_lr"])},
        ],
        weight_decay=wd,
    )
    sch_b = torch.optim.lr_scheduler.LambdaLR(opt_b, make_lr_lambda(warmup_epochs, epochs))

    best, bad = 0.0, 0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, tr_ld, crit, opt_b, device, True, mixup_alpha, cutmix_alpha,
            desc="train {}/{}".format(ep, epochs),
        )
        va_loss, va_acc = run_epoch(
            model, va_ld, crit, None, device, False,
            desc="val {}/{}".format(ep, epochs),
        )
        sch_b.step()
        lr0 = opt_b.param_groups[0]["lr"]
        print(
            "epoch", ep, "train", round(tr_acc, 4), round(tr_loss, 4),
            "val", round(va_acc, 4), round(va_loss, 4), "lr", round(lr0, 6),
        )
        if va_acc > best:
            best = va_acc
            bad = 0
            torch.save(unwrap(model).state_dict(), ckpt_path)
            print("  best val", round(best, 4), "->", ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print("early stop ep", ep)
                break

    print("{} best val {}".format(fold_label, round(best, 4)))
    return best


def train_full(f, df, img_dir, cls2idx, idx2cls, device, ckpt_path):
    """Train on the entire dataset for a fixed number of epochs (no early stopping)."""
    img_size = int(f.get("img_size", 224))
    bs = int(f["batch_size"])
    w = int(f["workers"])
    aug_strength = f.get("aug", "default")
    ds = ButterflyDataset(df.reset_index(drop=True), img_dir, get_supervised_transform(img_size, train=True, strength=aug_strength), cls2idx)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=w, pin_memory=True)
    print("full train {} samples".format(len(ds)))

    model = build_model(f, len(idx2cls), device)
    crit = nn.CrossEntropyLoss(label_smoothing=float(f.get("label_smoothing", 0.0)))
    mixup_alpha = float(f.get("mixup_alpha", 0.0))
    cutmix_alpha = float(f.get("cutmix_alpha", 0.0))

    probe_epochs = int(f.get("probe_epochs", 0))
    ckpt = (f.get("backbone_ckpt") or "").strip()
    if ckpt and probe_epochs > 0:
        print("linear probe", probe_epochs, "epochs")
        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(False)
        opt_a = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9
        )
        sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=probe_epochs)
        for ep in range(1, probe_epochs + 1):
            _, tr_acc = run_epoch(model, loader, crit, opt_a, device, True,
                                  desc="probe {}/{}".format(ep, probe_epochs))
            sch_a.step()
            print("probe", ep, "/", probe_epochs, "train acc", round(tr_acc, 4))
        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(True)

    epochs = int(f.get("full_train_epochs", f["epochs"]))
    warmup_epochs = int(f.get("warmup_epochs", 0))
    wd = float(f.get("weight_decay", 1e-4))
    opt_b = torch.optim.AdamW(
        [
            {"params": unwrap(model).backbone.parameters(), "lr": float(f["backbone_lr"])},
            {"params": unwrap(model).fc.parameters(), "lr": float(f["head_lr"])},
        ],
        weight_decay=wd,
    )
    sch_b = torch.optim.lr_scheduler.LambdaLR(opt_b, make_lr_lambda(warmup_epochs, epochs))

    print("full train {} epochs, warmup {}".format(epochs, warmup_epochs))
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, loader, crit, opt_b, device, True, mixup_alpha, cutmix_alpha,
            desc="train {}/{}".format(ep, epochs),
        )
        sch_b.step()
        lr0 = opt_b.param_groups[0]["lr"]
        print("epoch", ep, "train", round(tr_acc, 4), round(tr_loss, 4), "lr", round(lr0, 6))

    torch.save(unwrap(model).state_dict(), ckpt_path)
    print("saved ->", ckpt_path)


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
        print("device", device, "(no cuda)")

    cls2idx, idx2cls = build_label_encoder(data_root / "train.csv")
    with open(ckpt_dir / "label_encoder.json", "w") as fp:
        json.dump(idx2cls, fp)

    df = pd.read_csv(data_root / "train.csv")
    img_dir = data_root / "train_images" / "train_images"

    num_folds = int(f.get("folds", 1))

    if num_folds > 1:
        print("k-fold CV: {} folds".format(num_folds))
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_accs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df, df["TARGET"])):
            fold_label = "fold {}/{}".format(fold_idx + 1, num_folds)
            ckpt_path = ckpt_dir / "classifier_fold{}.pth".format(fold_idx)
            tr = df.iloc[tr_idx]
            va = df.iloc[va_idx]
            acc = train_fold(f, tr, va, img_dir, cls2idx, idx2cls, device, ckpt_path, fold_label)
            fold_accs.append(acc)
        print("CV results:", [round(a, 4) for a in fold_accs])
        print("mean val acc:", round(sum(fold_accs) / len(fold_accs), 4))
        # Save a symlink/copy of the best fold as classifier_best.pth for compatibility
        best_fold = max(range(num_folds), key=lambda i: fold_accs[i])
        best_src = ckpt_dir / "classifier_fold{}.pth".format(best_fold)
        best_dst = ckpt_dir / "classifier_best.pth"
        shutil.copy2(best_src, best_dst)
        print("best fold {} ({}) -> {}".format(best_fold, round(fold_accs[best_fold], 4), best_dst))
    elif float(f.get("val_split", 0.1)) == 0.0:
        train_full(f, df, img_dir, cls2idx, idx2cls, device, ckpt_dir / "classifier_best.pth")
    else:
        tr, va = train_test_split(
            df, test_size=float(f["val_split"]), stratify=df["TARGET"], random_state=42
        )
        train_fold(f, tr, va, img_dir, cls2idx, idx2cls, device,
                   ckpt_dir / "classifier_best.pth", "single split")


if __name__ == "__main__":
    main()
