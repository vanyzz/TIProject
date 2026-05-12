"""
Генерация аккомпанемента по записи бас-гитары.

Запуск:
    python src/inference.py --input my_bass.wav
    python src/inference.py --input my_bass.wav --output outputs/samples/result.wav

Если --input не указан — берётся трек из датасета для проверки.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import argparse
import torch
import musdb
import librosa
import numpy as np
import soundfile as sf

from models.encoder_decoder import AccompanimentModel

# ─── Настройки ────────────────────────────────────────────────
DEVICE     = "cpu"
MODEL_PATH = "models/final/model.pt"
DATA_ROOT  = "data/raw"
OUTPUT_DIR = "outputs/samples"

SR         = 22050
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
# ──────────────────────────────────────────────────────────────


def audio_to_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return (mel_db / 80.0).clip(-1.0, 1.0).astype(np.float32)


def mel_to_audio(mel_norm: np.ndarray) -> np.ndarray:
    """mel_norm [-1,1] -> аудио через быстрый Griffin-Lim."""
    mel_power   = librosa.db_to_power(mel_norm * 80.0)
    mel_filters = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    stft_approx = np.maximum(mel_filters.T @ mel_power, 0)
    audio = librosa.griffinlim(stft_approx, n_iter=32, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return audio.astype(np.float32)


def normalize(audio: np.ndarray, target_db: float = -6.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    return np.clip(audio * (10 ** (target_db / 20.0) / rms), -1.0, 1.0)


def save_wav(path: str, audio: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, SR)
    print(f"  Сохранено: {path}")


def run_model(bass_audio: np.ndarray) -> np.ndarray:
    """Прогоняет бас-аудио через модель, возвращает аккомпанемент."""
    model = AccompanimentModel(base_channels=32).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    print("Модель загружена.")

    mel = audio_to_mel(bass_audio)
    x   = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)

    print("Генерирую аккомпанемент...")
    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()

    print("Конвертирую в аудио...")
    return mel_to_audio(pred)


def from_file(input_path: str, output_path: str):
    """Режим: пользовательский wav-файл."""
    print(f"Загружаю бас: {input_path}")
    bass_audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
    if orig_sr != SR:
        bass_audio = librosa.resample(bass_audio, orig_sr=orig_sr, target_sr=SR)

    accomp = run_model(bass_audio)

    n = min(len(bass_audio), len(accomp))
    bass_audio = normalize(bass_audio[:n], -6.0)
    accomp     = normalize(accomp[:n],     -9.0)

    # Финальный микс: бас (громче) + аккомпанемент
    mixed = np.clip(bass_audio * 0.6 + accomp * 0.4, -1.0, 1.0)

    save_wav(output_path.replace(".wav", "_bass_only.wav"), bass_audio)
    save_wav(output_path.replace(".wav", "_accomp.wav"),    accomp)
    save_wav(output_path,                                   mixed)
    print(f"\nГотово!")
    print(f"  *_bass_only.wav — твоя запись баса")
    print(f"  *_accomp.wav    — сгенерированный аккомпанемент")
    print(f"  result.wav      — финальный микс (бас + аккомпанемент)")


def from_dataset():
    """Режим: проверка на треке из датасета."""
    db     = musdb.DB(root=DATA_ROOT, is_wav=False)
    tracks = db.load_mus_tracks(subsets="test")
    track  = tracks[0]
    print(f"Трек из датасета: {track.name}")

    track.chunk_start    = 30.0
    track.chunk_duration = 8.0

    def load(name):
        a = track.targets[name].audio.mean(axis=1).astype(np.float32)
        return librosa.resample(a, orig_sr=44100, target_sr=SR)

    bass_audio   = load("bass")
    real_accomp  = load("other")
    real_drums   = load("drums")
    real_full    = np.clip(real_accomp * 0.5 + real_drums * 0.5, -1.0, 1.0)

    gen_accomp = run_model(bass_audio)

    n = min(len(bass_audio), len(gen_accomp), len(real_full))
    bass_audio  = normalize(bass_audio[:n],  -6.0)
    real_full   = normalize(real_full[:n],   -9.0)
    gen_accomp  = normalize(gen_accomp[:n],  -9.0)

    mix_real = np.clip(bass_audio * 0.6 + real_full  * 0.4, -1.0, 1.0)
    mix_gen  = np.clip(bass_audio * 0.6 + gen_accomp * 0.4, -1.0, 1.0)

    save_wav(f"{OUTPUT_DIR}/01_bass_input.wav",      bass_audio)
    save_wav(f"{OUTPUT_DIR}/02_real_accomp.wav",     real_full)
    save_wav(f"{OUTPUT_DIR}/03_generated_accomp.wav",gen_accomp)
    save_wav(f"{OUTPUT_DIR}/04_mix_real.wav",        mix_real)
    save_wav(f"{OUTPUT_DIR}/05_mix_generated.wav",   mix_gen)

    print("\nСравни:")
    print("  04_mix_real.wav      — бас + реальный аккомпанемент")
    print("  05_mix_generated.wav — бас + сгенерированный аккомпанемент")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация аккомпанемента по басу")
    parser.add_argument("--input",  default=None, help="Путь к wav-файлу бас-гитары")
    parser.add_argument("--output", default="outputs/samples/result.wav")
    args = parser.parse_args()

    if args.input:
        from_file(args.input, args.output)
    else:
        print("Файл не указан — используем трек из датасета для проверки.")
        from_dataset()