"""
Датасет MIDI: гитара → фортепиано.
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


def midi_to_pianoroll_guitar_piano(path):
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
    return guitar_roll, piano_roll


class GuitarPianoDataset(Dataset):
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
            result = midi_to_pianoroll_guitar_piano(path)
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
                print(f"  файлов гитара+пианино: {found}, сегментов: {len(self.segments)}")
        print(f"[INFO] {split}: {len(self.segments)} сегментов из {found} файлов")
        if len(self.segments) == 0:
            raise ValueError("Не найдено ни одного подходящего сегмента")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        g, p = self.segments[idx]
        return torch.tensor(g).unsqueeze(0), torch.tensor(p).unsqueeze(0)