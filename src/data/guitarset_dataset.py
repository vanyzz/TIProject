"""
Датасет GuitarSet для обучения модели генерации аккомпанемента.

Структура файлов:
  data/guitarset/audio_mono-mic/   ← гитарные записи (*_comp_mic.wav = аккорды, *_solo_mic.wav = соло)
  data/guitarset/audio_mono-pickup_mix/  ← альтернативная запись

Пара для обучения:
  вход  = solo_mic  (одна гитарная линия)
  цель  = comp_mic  (полный аккорд/аккомпанемент)
"""

import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class GuitarSetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        sr: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        segment_sec: float = 2.0,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.segment_len = int(segment_sec * sr)

        audio_dir = os.path.join(root, "audio_mono-mic")
        # Все comp файлы (аккомпанемент) — это цель
        all_comp = sorted(glob.glob(os.path.join(audio_dir, "*_comp_mic.wav")))

        # Разбивка: игроки 00-04 = train, игрок 05 = val/test
        if split == "train":
            self.comp_files = [f for f in all_comp if os.path.basename(f).startswith(("00_", "01_", "02_", "03_"))]
        else:
            self.comp_files = [f for f in all_comp if os.path.basename(f).startswith(("04_", "05_"))]

        # Соответствующие solo файлы (вход)
        self.solo_files = [f.replace("_comp_mic.wav", "_solo_mic.wav") for f in self.comp_files]

        # Оставляем только пары где оба файла существуют
        pairs = [(s, c) for s, c in zip(self.solo_files, self.comp_files)
                 if os.path.exists(s) and os.path.exists(c)]
        self.solo_files  = [p[0] for p in pairs]
        self.comp_files  = [p[1] for p in pairs]

        print(f"[INFO] GuitarSet {split}: {len(self.comp_files)} треков")

    def _load_mel(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sr, mono=True)

        if len(y) < self.segment_len:
            y = np.pad(y, (0, self.segment_len - len(y)))

        max_start = len(y) - self.segment_len
        start = np.random.randint(0, max(1, max_start))
        y = y[start : start + self.segment_len].astype(np.float32)

        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length
        )
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db / 80.0).clip(-1.0, 1.0)
        return mel_norm.astype(np.float32)

    def __len__(self):
        return len(self.comp_files)

    def __getitem__(self, idx):
        mel_input  = torch.tensor(self._load_mel(self.solo_files[idx])).unsqueeze(0)
        mel_target = torch.tensor(self._load_mel(self.comp_files[idx])).unsqueeze(0)
        return mel_input, mel_target
