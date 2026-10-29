# ==============================================================================
# PASTE THIS ENTIRE BLOCK INTO A SINGLE GOOGLE COLAB CELL AND CLICK RUN
# (Make sure Runtime -> Change runtime type -> T4 GPU is selected)
# ==============================================================================
import os
import random
import subprocess
import sys
from pathlib import Path

# 1. Install dependencies automatically
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "kagglehub", "onnx", "onnxscript", "opencv-python-headless",
    "scikit-learn", "matplotlib", "tqdm"
])

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SEED = 42
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 2. Download MRL Dataset from Kaggle
dataset_root = Path(kagglehub.dataset_download("prasadvpatil/mrl-dataset"))
print(f"[INFO] Dataset downloaded to: {dataset_root}")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
train_samples, test_samples, all_samples = [], [], []

for img_path in dataset_root.rglob("*"):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue
    parts_lower = [p.lower() for p in img_path.parts]
    parent_lower = img_path.parent.name.lower()

    if "close" in parent_lower:
        label = 0  # Close-Eyes (Closed)
    elif "open" in parent_lower:
        label = 1  # Open-Eyes (Open)
    else:
        continue

    all_samples.append((str(img_path), label))
    if any("test" in p for p in parts_lower):
        test_samples.append((str(img_path), label))
    elif any("train" in p for p in parts_lower):
        train_samples.append((str(img_path), label))

if len(train_samples) > 0 and len(test_samples) > 0:
    labels = [s[1] for s in train_samples]
    train_list, val_list = train_test_split(
        train_samples, test_size=0.15, random_state=SEED, stratify=labels
    )
    test_list = test_samples
else:
    labels = [s[1] for s in all_samples]
    train_val, test_list = train_test_split(
        all_samples, test_size=0.15, random_state=SEED, stratify=labels
    )
    tv_labels = [s[1] for s in train_val]
    train_list, val_list = train_test_split(
        train_val, test_size=0.15, random_state=SEED, stratify=tv_labels
    )

print(f"[INFO] Total images: {len(all_samples)} | Train: {len(train_list)} | Val: {len(val_list)} | Test: {len(test_list)}")


# 3. Dataset with NoIR Grayscale + CLAHE Preprocessing
class MRLEyeDataset(Dataset):
    def __init__(self, samples, is_train: bool = False):
        self.samples = samples
        self.is_train = is_train
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def __len__(self):
        return len(self.samples)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.7:
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.90, 1.10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
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
        return torch.from_numpy(norm).unsqueeze(0), torch.tensor(label, dtype=torch.long)


train_loader = DataLoader(MRLEyeDataset(train_list, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(MRLEyeDataset(val_list, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(MRLEyeDataset(test_list, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# 4. Model Architecture (Matches app.py on Raspberry Pi 5)
class MRLEyeCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


model = MRLEyeCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 5. Train Loop
best_val_acc = 0.0
best_ckpt_path = "mrl_eye_model.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}", leave=False):
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

    train_acc = t_correct / t_total
    val_acc = v_correct / v_total

    saved_tag = ""
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_ckpt_path)
        saved_tag = " [*BEST*]"

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%{saved_tag}")

# 6. Evaluate Best Model on Test Set
model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x.to(device))
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_targets.extend(y.numpy())

print("\n=== Test Set Classification Report ===")
print(classification_report(all_targets, all_preds, target_names=["Close-Eyes (0)", "Open-Eyes (1)"], digits=4))
print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))

# 7. Export to Single-File ONNX & Auto-Download
onnx_path = "mrl_eye_model.onnx"
model_cpu = model.to("cpu").eval()
dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

try:
    torch.onnx.export(
        model_cpu, dummy_input, onnx_path,
        export_params=True, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["logits"], dynamo=False
    )
except TypeError:
    torch.onnx.export(
        model_cpu, dummy_input, onnx_path,
        export_params=True, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["logits"]
    )

# Verify OpenCV DNN compatibility
net = cv2.dnn.readNetFromONNX(onnx_path)
net.setInput(dummy_input.numpy())
_ = net.forward()
print(f"\n[SUCCESS] Exported and verified: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")

try:
    from google.colab import files
    files.download(onnx_path)
    files.download(best_ckpt_path)
except ImportError:
    pass

