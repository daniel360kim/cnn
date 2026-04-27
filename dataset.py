import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_label_encoder(csv_path):
    df = pd.read_csv(csv_path)
    idx2cls = sorted(df["TARGET"].unique())
    cls2idx = {c: i for i, c in enumerate(idx2cls)}
    return cls2idx, idx2cls


def get_simclr_transform(size=224, strong_blur=True):
    """One view for SimCLR. Call twice with strong_blur=True/False for asymmetric views."""
    k = int(0.1 * size) | 1
    blur_p = 0.5 if strong_blur else 0.1
    aug = [
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(k)], p=blur_p),
    ]
    try:
        aug.append(transforms.RandomSolarize(threshold=128, p=0.1))
    except AttributeError:
        pass  # torchvision < 0.11
    aug += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
    return transforms.Compose(aug)


def get_supervised_transform(size=224, train=True, strength="default"):
    if train:
        if strength == "strong":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(num_ops=2, magnitude=9),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_MEAN, std=_STD),
                    transforms.RandomErasing(p=0.4, scale=(0.02, 0.25)),
                ]
            )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.15),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomApply([transforms.GaussianBlur(5)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN, std=_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    val_size = int(size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(val_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


class SimCLRDataset(Dataset):
    def __init__(self, img_paths, aug_log_every=0):
        self.img_paths = img_paths
        # Asymmetric views: view 1 has strong blur, view 2 has weak blur
        self.transform1 = get_simclr_transform(strong_blur=True)
        self.transform2 = get_simclr_transform(strong_blur=False)
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
        v1 = self.transform1(img)
        if log:
            t2 = time.perf_counter()
        v2 = self.transform2(img)
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
    ds = SimCLRDataset(paths, aug_log_every=aug_log_every)
    # Update transforms to use the configured size
    ds.transform1 = get_simclr_transform(size=size, strong_blur=True)
    ds.transform2 = get_simclr_transform(size=size, strong_blur=False)
    return ds
