"""
Датасет MIDI v2: гитара → барабаны.
Улучшения:
  - Квантизация на сетку 1/16 при загрузке данных
  - Нормализация по типу инструмента (бочка/малый/тарелки)
"""

import os, glob, random
import torch
import numpy as np
import pretty_midi
from torch.utils.data import Dataset

SEGMENT_STEPS   = 128
FS              = 16   # 16 шагов/сек = 1/16 при 120 BPM
GUITAR_PROGRAMS = set(range(24, 32))

# GM стандарт — группы барабанных нот
KICK_PITCHES   = {35, 36}                          # бочка
SNARE_PITCHES  = {38, 40, 37}                      # малый барабан
HIHAT_PITCHES  = {42, 44, 46}                      # хэт
CYMBAL_PITCHES = {49, 51, 52, 53, 55, 57, 59}     # тарелки (крэш, райд)
TOM_PITCHES    = {41, 43, 45, 47, 48, 50}          # томы

# Веса для обучения — чем важнее инструмент, тем выше вес
PITCH_WEIGHTS = np.ones(128, dtype=np.float32)
for p in KICK_PITCHES:   PITCH_WEIGHTS[p] = 20.0  # бочка — очень важна
for p in SNARE_PITCHES:  PITCH_WEIGHTS[p] = 15.0  # малый — важен
for p in TOM_PITCHES:    PITCH_WEIGHTS[p] = 10.0  # томы — средне
for p in HIHAT_PITCHES:  PITCH_WEIGHTS[p] = 5.0   # хэт — менее важен
for p in CYMBAL_PITCHES: PITCH_WEIGHTS[p] = 3.0   # тарелки — наименее важны


def quantize_to_grid(roll, grid=1):
    """
    Квантизует piano roll на сетку grid шагов.
    Каждый удар становится ровно 1 шаг длиной (точечное событие).
    """
    T = roll.shape[1]
    quantized = np.zeros_like(roll)
    for pitch in range(128):
        prev_on = False
        for t in range(0, T, grid):
            # Берём максимум внутри ячейки сетки
            cell = roll[pitch, t:t+grid]
            if cell.max() > 0:
                if not prev_on:
                    quantized[pitch, t] = cell.max()  # только первый шаг
                    prev_on = True
            else:
                prev_on = False
    return quantized


def normalize_drums(drums_roll):
    """
    Нормализует барабанный roll:
    - Ограничивает количество тарелок (не более 1 в такт)
    - Убеждается что бочка и малый присутствуют если они есть в оригинале
    """
    T = drums_roll.shape[1]
    normalized = drums_roll.copy()

    # Для каждой группы тарелок — прореживаем до 1 удара каждые 4 шага
    cymbal_pitches = list(CYMBAL_PITCHES) + list(HIHAT_PITCHES)
    for t in range(T):
        active_cymbals = [p for p in cymbal_pitches if normalized[p, t] > 0]
        if len(active_cymbals) > 1:
            # Оставляем только один — с наибольшей вероятностью
            best = max(active_cymbals, key=lambda p: normalized[p, t])
            for p in active_cymbals:
                if p != best:
                    normalized[p, t] = 0

    return normalized


def midi_to_pianoroll_v2(path):
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None

    if pm.get_end_time() < 4.0:
        return None

    T = int(pm.get_end_time() * FS) + 1
    guitar_roll = np.zeros((128, T), dtype=np.float32)
    drums_roll  = np.zeros((128, T), dtype=np.float32)

    for inst in pm.instruments:
        if len(inst.notes) == 0:
            continue
        for note in inst.notes:
            t0 = int(note.start * FS)
            t1 = max(t0 + 1, int(note.end * FS))
            t1 = min(t1, T)
            if 0 <= t0 < T and note.pitch < 128:
                if inst.is_drum:
                    drums_roll[note.pitch, t0:t1] = 1.0
                elif inst.program in GUITAR_PROGRAMS:
                    guitar_roll[note.pitch, t0:t1] = 1.0

    if guitar_roll.max() < 0.01 or drums_roll.max() < 0.01:
        return None

    # Проверяем что есть хотя бы бочка или малый
    has_kick  = any(drums_roll[p].max() > 0 for p in KICK_PITCHES)
    has_snare = any(drums_roll[p].max() > 0 for p in SNARE_PITCHES)
    if not has_kick and not has_snare:
        return None  # пропускаем треки только с тарелками

    # Квантизуем барабаны на сетку 1/16
    drums_roll = quantize_to_grid(drums_roll, grid=1)
    # Нормализуем (прореживаем тарелки)
    drums_roll = normalize_drums(drums_roll)

    return guitar_roll, drums_roll


class GuitarDrumsDatasetV2(Dataset):
    def __init__(self, root, split="train", max_files=5000):
        self.segments = []

        all_files  = glob.glob(os.path.join(root, "**", "*.mid"),  recursive=True)
        all_files += glob.glob(os.path.join(root, "**", "*.midi"), recursive=True)

        random.seed(42)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * 0.9)
        files = all_files[:split_idx] if split == "train" else all_files[split_idx:]
        files = files[:max_files]

        print(f"[INFO] Обрабатываю до {len(files)} MIDI файлов ({split})...")
        found = 0
        for path in files:
            result = midi_to_pianoroll_v2(path)
            if result is None:
                continue
            guitar, drums = result
            T = guitar.shape[1]
            for start in range(0, T - SEGMENT_STEPS, SEGMENT_STEPS // 2):
                g = guitar[:, start:start + SEGMENT_STEPS]
                d = drums[:,  start:start + SEGMENT_STEPS]
                if g.max() > 0.01 and d.max() > 0.01:
                    self.segments.append((g, d))
            found += 1
            if found % 200 == 0:
                print(f"  файлов: {found}, сегментов: {len(self.segments)}")

        print(f"[INFO] {split}: {len(self.segments)} сегментов из {found} файлов")
        if len(self.segments) == 0:
            raise ValueError("Не найдено ни одного подходящего сегмента")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        g, d = self.segments[idx]
        return torch.tensor(g).unsqueeze(0), torch.tensor(d).unsqueeze(0)
