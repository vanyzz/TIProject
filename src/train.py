"""
Скрипт обучения модели генерации аккомпанемента.
Поддерживает CPU, CUDA и AMD GPU через DirectML.

Запуск из папки TIProject:
    python src/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from data.dataset import MUSDBDataset
from data.guitarset_dataset import GuitarSetDataset
from models.encoder_decoder import AccompanimentModel

# ─── Устройство ───────────────────────────────────────────────
def get_device():
    try:
        import torch_directml
        print("Используется AMD GPU через DirectML")
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        print("Используется NVIDIA CUDA GPU")
        return torch.device("cuda")
    print("Используется CPU")
    return torch.device("cpu")

# ─── Настройки ────────────────────────────────────────────────
DEVICE     = get_device()
MUSDB_ROOT = "data/raw"
GUITAR_ROOT= "data/guitarset"
CKPT_DIR   = "models/checkpoints"
FINAL_PATH = "models/final/model.pt"

EPOCHS     = 50
BATCH_SIZE = 8
LR         = 5e-4
SAVE_EVERY = 10
# ──────────────────────────────────────────────────────────────


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_PATH), exist_ok=True)

    # Данные — объединяем MUSDB18 + GuitarSet
    musdb_train = MUSDBDataset(MUSDB_ROOT, split="train")
    musdb_val   = MUSDBDataset(MUSDB_ROOT, split="test")

    try:
        guitar_train = GuitarSetDataset(GUITAR_ROOT, split="train")
        guitar_val   = GuitarSetDataset(GUITAR_ROOT, split="val")
        train_ds = ConcatDataset([musdb_train, guitar_train])
        val_ds   = ConcatDataset([musdb_val,   guitar_val])
        print(f"Датасет: MUSDB18 + GuitarSet = {len(train_ds)} train / {len(val_ds)} val сэмплов")
    except Exception as e:
        print(f"GuitarSet не загружен ({e}), используем только MUSDB18")
        train_ds = musdb_train
        val_ds   = musdb_val

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Модель
    model     = AccompanimentModel(base_channels=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total_params:,}")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred  = model(x)
            min_t = min(pred.shape[-1], y.shape[-1])
            loss  = criterion(pred[..., :min_t], y[..., :min_t])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred  = model(x)
                min_t = min(pred.shape[-1], y.shape[-1])
                val_loss += criterion(pred[..., :min_t], y[..., :min_t]).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if epoch % SAVE_EVERY == 0:
            ckpt = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt)
            print(f"  Чекпоинт: {ckpt}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), FINAL_PATH)
            print(f"  Лучшая модель (val={val_loss:.4f})")

    print(f"\nОбучение завершено! Лучший val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
