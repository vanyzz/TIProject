"""
Обучение GRU модели: гитара → барабаны.
Запуск: python src/train_guitar_drums.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

from data.midi_dataset_guitar_drums import GuitarDrumsDataset
from models.lstm_model import LSTMAccompanimentModel

DEVICE        = torch.device("cpu")
DATA_ROOT     = "data/lmd/lmd_matched"
CKPT_DIR      = "models/guitar_drums_checkpoints"
FINAL_PATH    = "models/guitar_drums_final/model.pt"

EPOCHS        = 30
BATCH_SIZE    = 64
LR            = 1e-3
SAVE_EVERY    = 5
TRAIN_SAMPLES = 8000
VAL_SAMPLES   = 2000


class FocalMSELoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        weights = 1.0 + (self.pos_weight - 1.0) * target
        return (weights * (pred - target) ** 2).mean()


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_PATH), exist_ok=True)

    print("Загружаю датасет (гитара → барабаны)...")
    train_ds = GuitarDrumsDataset(DATA_ROOT, split="train", max_files=5000)
    val_ds   = GuitarDrumsDataset(DATA_ROOT, split="val",   max_files=1000)

    train_idx = random.sample(range(len(train_ds)), min(TRAIN_SAMPLES, len(train_ds)))
    val_idx   = random.sample(range(len(val_ds)),   min(VAL_SAMPLES,   len(val_ds)))
    train_sub = Subset(train_ds, train_idx)
    val_sub   = Subset(val_ds,   val_idx)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = LSTMAccompanimentModel(hidden_size=128, num_layers=1).to(DEVICE)
    criterion = FocalMSELoss(pos_weight=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"Train: {len(train_sub)} | Val: {len(val_sub)}")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            pred  = model(x)
            min_t = min(pred.shape[-1], y.shape[-1])
            loss  = criterion(pred[..., :min_t], y[..., :min_t])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred  = model(x)
                min_t = min(pred.shape[-1], y.shape[-1])
                val_loss += criterion(pred[..., :min_t], y[..., :min_t]).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if epoch % SAVE_EVERY == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), FINAL_PATH)
            print(f"  Лучшая модель (val={val_loss:.4f})")

    print(f"\nГотово! Лучший val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()