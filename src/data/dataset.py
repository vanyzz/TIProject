"""
Загрузка датасета MUSDB18 из .stem.mp4 файлов.

Использование:
    dataset = MUSDBDataset(root="data/raw", split="train")
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)
"""

import torch
import musdb
import librosa
import numpy as np
from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        sr: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        segment_sec: float = 2.0,
        input_stem: str = "bass",
        target_stem: str = "other",
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.segment_len = int(segment_sec * sr)
        self.input_stem = input_stem
        self.target_stem = target_stem

        db = musdb.DB(root=root, is_wav=False)
        subset = "train" if split == "train" else "test"
        self.tracks = db.load_mus_tracks(subsets=subset)
        print(f"[INFO] {split}: загружено {len(self.tracks)} треков")

    def _audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Стерео -> моно -> ресэмплинг -> мел-спектрограмма [-1, 1]."""
        # Стерео -> моно
        if audio.ndim == 2:
            y = audio.mean(axis=1)
        else:
            y = audio

        y = y.astype(np.float32)

        # Ресэмплинг 44100 -> 22050
        if len(y) > 0:
            y = librosa.resample(y, orig_sr=44100, target_sr=self.sr)

        # Если трек короткий — дополняем нулями
        if len(y) < self.segment_len:
            y = np.pad(y, (0, self.segment_len - len(y)))

        # Случайный сегмент
        max_start = len(y) - self.segment_len
        start = np.random.randint(0, max(1, max_start))
        y = y[start : start + self.segment_len]

        # Мел-спектрограмма
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db / 80.0).clip(-1.0, 1.0)
        return mel_norm.astype(np.float32)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]

        # Говорим musdb какой кусок грузить
        chunk_dur = self.segment_len / self.sr
        max_start = max(0.0, track.duration - chunk_dur)
        track.chunk_start    = float(np.random.uniform(0, max_start)) if max_start > 0 else 0.0
        track.chunk_duration = chunk_dur

        audio_input  = track.targets[self.input_stem].audio
        audio_target = track.targets[self.target_stem].audio

        mel_input  = torch.tensor(self._audio_to_mel(audio_input)).unsqueeze(0)
        mel_target = torch.tensor(self._audio_to_mel(audio_target)).unsqueeze(0)

        return mel_input, mel_target