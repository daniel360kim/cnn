import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_label_encoder(csv_path):
    df = pd.read_csv(csv_path)
    idx2cls = sorted(df["TARGET"].unique())
    cls2idx = {c: i for i, c in enumerate(idx2cls)}
    return cls2idx, idx2cls


def get_simclr_transform(size=224):
    k = int(0.1 * size) | 1
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(k)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_supervised_transform(size=224, train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SimCLRDataset(Dataset):
    def __init__(self, img_paths, transform, aug_log_every=0):
        self.img_paths = img_paths
        self.transform = transform
        self.aug_log_every = aug_log_every
        self._aug_n = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        log = False
        if self.aug_log_every:
            self._aug_n += 1
            log = self._aug_n % self.aug_log_every == 0
        if log:
            t0 = time.perf_counter()
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if log:
            t1 = time.perf_counter()
        v1 = self.transform(img)
        if log:
            t2 = time.perf_counter()
        v2 = self.transform(img)
        if log:
            t3 = time.perf_counter()
            print(
                "[aug] pid=%s n=%s open_ms=%.1f v1_ms=%.1f v2_ms=%.1f total_ms=%.1f"
                % (
                    os.getpid(),
                    self._aug_n,
                    (t1 - t0) * 1000,
                    (t2 - t1) * 1000,
                    (t3 - t2) * 1000,
                    (t3 - t0) * 1000,
                ),
                flush=True,
            )
        return v1, v2


class ButterflyDataset(Dataset):
    def __init__(self, df, img_dir, transform, cls2idx):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.cls2idx = cls2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["file_name"]).convert("RGB")
        return self.transform(img), self.cls2idx[row["TARGET"]]


class TestDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_paths = sorted(Path(img_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path.stem


def build_simclr_dataset(data_root, size=224, aug_log_every=0):
    train_dir = Path(data_root) / "train_images" / "train_images"
    test_dir = Path(data_root) / "test_images" / "test_images"
    paths = sorted(train_dir.glob("*.jpg")) + sorted(test_dir.glob("*.jpg"))
    return SimCLRDataset(paths, get_simclr_transform(size), aug_log_every=aug_log_every)
