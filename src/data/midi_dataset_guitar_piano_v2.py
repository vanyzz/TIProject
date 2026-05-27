"""
Датасет MIDI v2: гитара → фортепиано.
Улучшения:
  - Квантизация нот на сетку 1/16
  - Ограничение одновременных нот (максимум 3 — трезвучие)
  - Фильтрация треков без явной мелодической линии
"""

import os, glob, random
import torch
import numpy as np
import pretty_midi
from torch.utils.data import Dataset

SEGMENT_STEPS   = 128
FS              = 16
GUITAR_PROGRAMS = set(range(24, 32))
PIANO_PROGRAMS  = set(range(0, 8))

MAX_SIMULTANEOUS_PIANO = 3  # максимум нот одновременно (трезвучие)


def quantize_piano_roll(roll, grid=1):
    """
    Квантизует piano roll на сетку grid шагов.
    В отличие от барабанов — сохраняем длину нот, но начало привязываем к сетке.
    """
    if grid <= 1:
        return roll
    T = roll.shape[1]
    quantized = np.zeros_like(roll)
    for pitch in range(128):
        in_note = False
        note_val = 0.0
        for t in range(T):
            if roll[pitch, t] > 0 and not in_note:
                # Начало ноты — привязываем к ближайшей сетке
                snapped_t = round(t / grid) * grid
                snapped_t = min(snapped_t, T - 1)
                note_val = roll[pitch, t]
                in_note = True
                quantized[pitch, snapped_t] = note_val
            elif roll[pitch, t] > 0 and in_note:
                quantized[pitch, t] = note_val  # продолжаем ноту
            elif roll[pitch, t] == 0 and in_note:
                in_note = False  # конец ноты
    return quantized


def limit_simultaneous_notes(roll, max_notes=3):
    """
    Ограничивает количество одновременно звучащих нот.
    Оставляет только max_notes наиболее вероятных.
    """
    T = roll.shape[1]
    limited = roll.copy()
    for t in range(T):
        active = np.where(limited[:, t] > 0)[0]
        if len(active) > max_notes:
            # Оставляем средний диапазон — исключаем крайние высокие/низкие
            # Сортируем по питчу и берём средние max_notes нот
            sorted_pitches = sorted(active)
            remove_count = len(active) - max_notes
            # Убираем самые крайние (нижние и верхние)
            to_remove = []
            low, high = 0, len(sorted_pitches) - 1
            while len(to_remove) < remove_count:
                if low <= high:
                    to_remove.append(sorted_pitches[low])
                    low += 1
                    if len(to_remove) < remove_count and low <= high:
                        to_remove.append(sorted_pitches[high])
                        high -= 1
            for p in to_remove:
                limited[p, t] = 0
    return limited


def midi_to_pianoroll_piano_v2(path):
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None

    if pm.get_end_time() < 4.0:
        return None

    T = int(pm.get_end_time() * FS) + 1
    guitar_roll = np.zeros((128, T), dtype=np.float32)
    piano_roll  = np.zeros((128, T), dtype=np.float32)

    for inst in pm.instruments:
        if len(inst.notes) == 0 or inst.is_drum:
            continue
        for note in inst.notes:
            t0 = int(note.start * FS)
            t1 = max(t0 + 1, int(note.end * FS))
            t1 = min(t1, T)
            if 0 <= t0 < T and note.pitch < 128:
                if inst.program in GUITAR_PROGRAMS:
                    guitar_roll[note.pitch, t0:t1] = 1.0
                elif inst.program in PIANO_PROGRAMS:
                    piano_roll[note.pitch, t0:t1] = 1.0

    if guitar_roll.max() < 0.01 or piano_roll.max() < 0.01:
        return None

    # Квантизуем пианино на сетку 1/16
    piano_roll = quantize_piano_roll(piano_roll, grid=1)
    # Ограничиваем одновременные ноты
    piano_roll = limit_simultaneous_notes(piano_roll, max_notes=MAX_SIMULTANEOUS_PIANO)

    return guitar_roll, piano_roll


class GuitarPianoDatasetV2(Dataset):
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
            result = midi_to_pianoroll_piano_v2(path)
            if result is None:
                continue
            guitar, piano = result
            T = guitar.shape[1]
            for start in range(0, T - SEGMENT_STEPS, SEGMENT_STEPS // 2):
                g = guitar[:, start:start + SEGMENT_STEPS]
                p = piano[:,  start:start + SEGMENT_STEPS]
                if g.max() > 0.01 and p.max() > 0.01:
                    self.segments.append((g, p))
            found += 1
            if found % 200 == 0:
                print(f"  файлов: {found}, сегментов: {len(self.segments)}")

        print(f"[INFO] {split}: {len(self.segments)} сегментов из {found} файлов")
        if len(self.segments) == 0:
            raise ValueError("Не найдено ни одного подходящего сегмента")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        g, p = self.segments[idx]
        return torch.tensor(g).unsqueeze(0), torch.tensor(p).unsqueeze(0)