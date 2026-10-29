"""
Standalone Training Script for the Kaggle MRL Eye Dataset
Dataset: https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

Can be run either in Google Colab (`!python train.py`) or locally (`python train.py`).
Automatically downloads `prasadvpatil/mrl-dataset` via `kagglehub` if `--data-dir` is not specified,
trains the NoIR-optimized `MRLEyeCNN`, and exports `mrl_eye_model.onnx` and `mrl_eye_model.pt`.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app import build_pytorch_model

IMG_SIZE = 64
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class MRLEyeDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], is_train: bool = False):
        self.samples = samples
        self.is_train = is_train
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.7:
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.90, 1.10)
            h, w = img.shape[:2]
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.6:
            alpha = random.uniform(0.75, 1.25)
            beta = random.uniform(-20, 20)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        if self.is_train:
            gray = self._augment(gray)

        eq = self.clahe.apply(gray)
        resized = cv2.resize(eq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(norm).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


def collect_mrl_samples(dataset_root: Path, seed: int = 42):
    train_samples, test_samples, all_samples = [], [], []
    for img_path in dataset_root.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts_lower = [p.lower() for p in img_path.parts]
        parent_lower = img_path.parent.name.lower()

        if "close" in parent_lower:
            label = 0
        elif "open" in parent_lower:
            label = 1
        else:
            continue

        all_samples.append((str(img_path), label))
        if any("test" in p for p in parts_lower):
            test_samples.append((str(img_path), label))
        elif any("train" in p for p in parts_lower):
            train_samples.append((str(img_path), label))

    if not all_samples:
        raise RuntimeError(f"No Open/Close eye images found under: {dataset_root}")

    if train_samples and test_samples:
        labels = [s[1] for s in train_samples]
        train_list, val_list = train_test_split(
            train_samples, test_size=0.15, random_state=seed, stratify=labels
        )
        test_list = test_samples
    else:
        labels = [s[1] for s in all_samples]
        train_val, test_list = train_test_split(
            all_samples, test_size=0.15, random_state=seed, stratify=labels
        )
        tv_labels = [s[1] for s in train_val]
        train_list, val_list = train_test_split(
            train_val, test_size=0.15, random_state=seed, stratify=tv_labels
        )

    return train_list, val_list, test_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRL Eye CNN for RPi 5 NoIR Drowsiness Detection")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Local path to MRL dataset folder. If omitted, downloads prasadvpatil/mrl-dataset via kagglehub.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--onnx-out", type=str, default="mrl_eye_model.onnx", help="Output ONNX file path")
    parser.add_argument("--pt-out", type=str, default="mrl_eye_model.pt", help="Output PyTorch file path")
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.data_dir:
        dataset_root = Path(args.data_dir)
    else:
        import kagglehub

        print("[INFO] Downloading prasadvpatil/mrl-dataset via kagglehub...")
        dataset_root = Path(kagglehub.dataset_download("prasadvpatil/mrl-dataset"))

    print(f"[INFO] Using dataset root: {dataset_root}")
    train_list, val_list, test_list = collect_mrl_samples(dataset_root, seed=seed)
    print(f"[INFO] Train: {len(train_list)} | Val: {len(val_list)} | Test: {len(test_list)}")

    train_loader = DataLoader(MRLEyeDataset(train_list, is_train=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MRLEyeDataset(val_list, is_train=False), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MRLEyeDataset(test_list, is_train=False), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    model = build_pytorch_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            t_correct += (logits.argmax(dim=1) == y).sum().item()
            t_total += x.size(0)

        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (logits.argmax(dim=1) == y).sum().item()
                v_total += x.size(0)

        val_acc = v_correct / v_total
        tag = ""
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.pt_out)
            tag = " [*BEST*]"

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Acc: {t_correct / t_total * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}%{tag}"
        )

    # Test evaluation
    model.load_state_dict(torch.load(args.pt_out, map_location=device, weights_only=True))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.numpy())

    print("\n=== Test Classification Report ===")
    print(classification_report(targets, preds, target_names=["Close-Eyes (0)", "Open-Eyes (1)"], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))

    # Export ONNX
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    try:
        torch.onnx.export(
            model_cpu,
            dummy,
            args.onnx_out,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model_cpu,
            dummy,
            args.onnx_out,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
        )

    print(f"[SUCCESS] Exported ONNX model to: {args.onnx_out} ({os.path.getsize(args.onnx_out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

