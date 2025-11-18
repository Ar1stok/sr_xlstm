from __future__ import annotations

import logging

import librosa
from numpy import ndarray
from datasets import Audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _compute_melspectrogram(audio: Audio) -> ndarray:
    """
    Convert audiofile to melspectrogram using librosa

    Parameters
    ----------
    audio: Audio
        audiofile with type Audio from datasets
    
    Returns:
    --------
    ndarray:
        melspectrogram like numpy array
    """
    signal = audio["array"]
    sr = audio["sampling_rate"]
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        win_length=400,
        n_fft=512,
        hop_length=160,
        fmin=50,
        fmax=3500,
        n_mels=32,
        )
    return mel_spectrogram


def add_melspectrogram(example):
    example["input_ids"] = _compute_melspectrogram(example["audio"])
    return example


class DataCollatorxLSTM:
    def __init__(self):
        pass