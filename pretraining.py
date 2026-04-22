#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path

import gpu_env

gpu_env.set_visible_gpus()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import cfg
from dataset import build_simclr_dataset
from models import ResNet18, SimCLRModel


def nt_xent(z1, z2, t):
    n = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.mm(z, z.T) / t
    sim.fill_diagonal_(-float("inf"))
    dev = z.device
    y = torch.cat([torch.arange(n, 2 * n, device=dev), torch.arange(n, device=dev)])
    return F.cross_entropy(sim, y)


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def cosine_lr(opt, base_lr, step, total, warmup):
    if step < warmup:
        lr = base_lr * (step + 1) / warmup
    else:
        p = (step - warmup) / max(total - warmup, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * p))
    set_lr(opt, lr)
    return lr


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def _worker_init(_):
    # Avoid worker processes each spawning many BLAS threads (CPU oversubscription → slow dataloader).
    import torch as _t

    _t.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=cfg.ROOT / "config.toml")
    args = ap.parse_args()

    c = cfg.load(args.config)
    p = c["pretrain"]
    paths = c["paths"]
    data_root = Path(paths["data_root"])
    ckpt_dir = Path(paths["ckpt_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = cfg.ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # benchmark + DataParallel threads -> CUDNN_STATUS_NOT_INITIALIZED on some setups
        torch.backends.cudnn.benchmark = torch.cuda.device_count() == 1
        print("device", device, torch.cuda.get_device_name(0))
        # Ampere (SM80+) and newer: TF32 speeds convs and the NT-Xent matmul with negligible impact here.
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 matmul/cudnn enabled (device sm_>=80)")
    else:
        print("device", device, "(no cuda — check driver and CUDA_VISIBLE_DEVICES)")

    log_aug = int(p.get("log_aug_every", 0))
    ds = build_simclr_dataset(data_root=data_root, aug_log_every=log_aug)
    w = int(p["workers"])
    kw = dict(
        batch_size=int(p["batch_size"]),
        shuffle=True,
        num_workers=w,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if w > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, int(p.get("prefetch_factor", 4)))
        kw["worker_init_fn"] = _worker_init
    loader = DataLoader(ds, **kw)
    print(len(ds), "images", len(loader), "batches/epoch")

    m = SimCLRModel(ResNet18())
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    m = m.to(device)
    if (
        device.type == "cuda"
        and bool(p.get("torch_compile", False))
        and torch.cuda.device_count() == 1
    ):
        try:
            m = torch.compile(m)
            print("torch.compile enabled (single-GPU)")
        except Exception as ex:
            print("torch.compile skipped:", ex)
    opt = torch.optim.SGD(m.parameters(), lr=float(p["lr"]), momentum=0.9, weight_decay=1e-4)
    use_amp = device.type == "cuda" and bool(p.get("use_amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
    if use_amp:
        print("AMP (fp16) enabled for pretrain")

    epochs = int(p["epochs"])
    total = epochs * len(loader)
    warm = int(p["warmup_epochs"]) * len(loader)
    step = 0
    first_step = True
    # Avoid loss.item() / tqdm postfix every step — GPU sync stalls overlap with the next batch.
    pbar_refresh = max(1, int(p.get("pbar_refresh_every", 20)))

    with tqdm(
        total=total,
        desc="pretrain",
        dynamic_ncols=True,
        mininterval=0.5,
        miniters=max(10, total // 200),
        # Default smoothing keeps the first (slow) step in the ETA for a long time.
        smoothing=0.45,
    ) as pbar:
        for ep in range(1, epochs + 1):
            m.train()
            loss_sum_t = torch.zeros((), device=device)
            t_ep = time.time()
            for x1, x2 in loader:
                t0 = time.time()
                lr = cosine_lr(opt, float(p["lr"]), step, total, warm)
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                n = x1.size(0)
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        z = m(torch.cat([x1, x2], dim=0))
                        z1, z2 = z[:n], z[n:]
                        loss = nt_xent(z1, z2, float(p["temperature"]))
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    z = m(torch.cat([x1, x2], dim=0))
                    z1, z2 = z[:n], z[n:]
                    loss = nt_xent(z1, z2, float(p["temperature"]))
                    loss.backward()
                    opt.step()
                loss_sum_t += loss.detach()
                step += 1
                pbar.update(1)
                if step % pbar_refresh == 0 or step == 1:
                    pbar.set_postfix(
                        ep="{}/{}".format(ep, epochs),
                        loss="{:.4f}".format(loss.detach().item()),
                        lr="{:.6f}".format(lr),
                    )
                if first_step:
                    first_step = False
                    print("first step {:.1f}s".format(time.time() - t0))
            avg = float(loss_sum_t / len(loader))
            print("epoch", ep, "/", epochs, "loss", round(avg, 4), "sec", int(time.time() - t_ep))

            se = int(p["save_every"])
            if se and ep % se == 0:
                out = ckpt_dir / "simclr_epoch{:04d}.pth".format(ep)
                torch.save(unwrap(m).backbone.state_dict(), out)
                print("saved", out)

    final = ckpt_dir / "simclr_final.pth"
    torch.save(unwrap(m).backbone.state_dict(), final)
    print("done", final)


if __name__ == "__main__":
    main()
