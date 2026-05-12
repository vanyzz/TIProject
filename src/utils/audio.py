import numpy as np
import librosa
import soundfile as sf


def mel_to_audio(
    mel_norm: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_iter: int = 64,
) -> np.ndarray:
    """
    Конвертирует нормализованную мел-спектрограмму [-1, 1] обратно в аудио.
    Использует алгоритм Griffin-Lim для оценки фазы.

    Args:
        mel_norm:   спектрограмма формата (n_mels, T), значения в [-1, 1]
        sr:         частота дискретизации
        hop_length: шаг окна
        n_iter:     итерации Griffin-Lim (больше = чище, но медленнее)

    Returns:
        audio: np.ndarray формата (N,)
    """
    # Денормализация: [-1, 1] → dB → power
    mel_db = mel_norm * 80.0
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=sr,
        hop_length=hop_length,
        n_iter=n_iter,
    )
    return audio.astype(np.float32)


def mix_tracks(tracks: list, weights: list = None) -> np.ndarray:
    """
    Смешивает список аудио-дорожек с заданными весами.

    Args:
        tracks:  список np.ndarray (могут быть разной длины)
        weights: коэффициенты громкости; по умолчанию равные доли

    Returns:
        mixed: np.ndarray, значения обрезаны в [-1, 1]
    """
    if not tracks:
        raise ValueError("Список треков пуст")
    max_len = max(len(t) for t in tracks)
    if weights is None:
        weights = [1.0 / len(tracks)] * len(tracks)
    if len(weights) != len(tracks):
        raise ValueError("Количество весов не совпадает с количеством треков")

    mixed = np.zeros(max_len, dtype=np.float32)
    for track, w in zip(tracks, weights):
        padded = np.pad(track.astype(np.float32), (0, max_len - len(track)))
        mixed += padded * w

    return np.clip(mixed, -1.0, 1.0)


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Нормализует громкость до заданного уровня в dB.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return (audio * (target_rms / rms)).clip(-1.0, 1.0)


def save_audio(audio: np.ndarray, path: str, sr: int = 22050):
    """Сохраняет аудио в .wav файл."""
    sf.write(path, audio, sr)
    print(f"[OK] Сохранено: {path}")
