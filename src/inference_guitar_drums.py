"""
Генерация барабанов по гитарной партии (новая модель).

Запуск:
    python src/inference_guitar_drums.py --demo
    python src/inference_guitar_drums.py --input my_guitar.mid
"""

import sys, os, argparse, glob, random
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pretty_midi

from models.lstm_model import LSTMAccompanimentModel
from data.midi_dataset_guitar_drums import FS, SEGMENT_STEPS, GUITAR_PROGRAMS

DEVICE     = torch.device("cpu")
MODEL_PATH = "models/guitar_drums_final/model.pt"
OUTPUT_DIR = "outputs/midi"
THRESHOLD  = 0.5


def load_model():
    model = LSTMAccompanimentModel(hidden_size=128, num_layers=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    print("Модель загружена.")
    return model


def notes_to_roll(pm, T):
    """Извлекает только гитарный piano roll."""
    roll = np.zeros((128, T), dtype=np.float32)
    for inst in pm.instruments:
        if not inst.is_drum and inst.program in GUITAR_PROGRAMS:
            for note in inst.notes:
                t0 = int(note.start * FS)
                t1 = max(t0 + 1, int(note.end * FS))
                if t0 < T:
                    roll[note.pitch, t0:min(t1, T)] = 1.0
    return roll


def generate_drums(guitar_roll, model):
    T = guitar_roll.shape[1]
    result = np.zeros((128, T), dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    step = SEGMENT_STEPS // 2
    for start in range(0, T, step):
        seg = guitar_roll[:, start:start + SEGMENT_STEPS]
        if seg.shape[1] < SEGMENT_STEPS:
            seg = np.pad(seg, ((0, 0), (0, SEGMENT_STEPS - seg.shape[1])))
        x = torch.tensor(seg).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).squeeze().numpy()
        w = min(SEGMENT_STEPS, T - start)
        result[:, start:start + w] += pred[:, :w]
        counts[start:start + w] += 1
    return result / np.maximum(counts, 1)


def roll_to_drums(roll):
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Generated Drums")
    dt = 1.0 / FS
    for pitch in range(128):
        active, start = False, 0.0
        for t in range(roll.shape[1]):
            on = roll[pitch, t] > THRESHOLD
            if on and not active:
                start, active = t * dt, True
            elif not on and active:
                inst.notes.append(pretty_midi.Note(80, pitch, start, t * dt))
                active = False
        if active:
            inst.notes.append(pretty_midi.Note(80, pitch, start, roll.shape[1] * dt))
    return inst


def process(midi_path, output_path, model):
    print(f"Файл: {midi_path}")
    pm = pretty_midi.PrettyMIDI(midi_path)
    T  = int(pm.get_end_time() * FS) + 1

    guitar_roll = notes_to_roll(pm, T)
    if guitar_roll.max() < 0.01:
        print("  Гитара не найдена — используем все не-барабанные инструменты")
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    t0 = int(note.start * FS)
                    t1 = max(t0 + 1, int(note.end * FS))
                    if t0 < T:
                        guitar_roll[note.pitch, t0:min(t1, T)] = 1.0

    print("Генерирую барабаны...")
    drums_roll = generate_drums(guitar_roll, model)

    # Сохраняем два файла для сравнения
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Только гитара (без барабанов)
    pm_no_drums = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())
    for inst in pm.instruments:
        if not inst.is_drum:
            pm_no_drums.instruments.append(inst)
    pm_no_drums.write(os.path.join(OUTPUT_DIR, "guitar_only.mid"))

    # Гитара + оригинальные барабаны
    pm_orig = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())
    for inst in pm.instruments:
        pm_orig.instruments.append(inst)
    pm_orig.write(os.path.join(OUTPUT_DIR, "guitar_original_drums.mid"))

    # Гитара + сгенерированные барабаны
    drums_inst = roll_to_drums(drums_roll)
    pm_gen = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())
    for inst in pm.instruments:
        if not inst.is_drum:
            pm_gen.instruments.append(inst)
    pm_gen.instruments.append(drums_inst)
    pm_gen.write(output_path)

    print(f"\nСохранено:")
    print(f"  guitar_only.mid            — только гитара (вход)")
    print(f"  guitar_original_drums.mid  — гитара + оригинальные барабаны (эталон)")
    print(f"  {os.path.basename(output_path)}     — гитара + сгенерированные барабаны")
    print(f"  Нот в барабанах: {len(drums_inst.notes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=None)
    parser.add_argument("--output", default="outputs/midi/guitar_generated_drums.mid")
    parser.add_argument("--demo",   action="store_true")
    args = parser.parse_args()

    model = load_model()

    if args.demo:
        # Ищем файл с гитарой и барабанами
        files = glob.glob("data/lmd/lmd_matched/**/*.mid", recursive=True)
        random.shuffle(files)
        for f in files[:100]:
            try:
                pm = pretty_midi.PrettyMIDI(f)
                has_guitar = any(i.program in GUITAR_PROGRAMS and not i.is_drum
                                 and len(i.notes) > 10 for i in pm.instruments)
                has_drums  = any(i.is_drum and len(i.notes) > 10 for i in pm.instruments)
                if has_guitar and has_drums:
                    process(f, "outputs/midi/guitar_generated_drums.mid", model)
                    break
            except Exception:
                continue
    elif args.input:
        process(args.input, args.output, model)
    else:
        print("Укажи --input file.mid или --demo")