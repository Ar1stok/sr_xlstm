""" Data Process: Convert audiofile to melspectrogram """
from __future__ import annotations

import logging

import torch
import torchaudio.transforms as T
from datasets import Audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _compute_melspectrogram(audio: Audio) -> torch.Tensor:
    """
    Convert audiofile to melspectrogram using torchaudio

    Parameters
    ----------
    audio: Audio
        Audiofile with type Audio from datasets
    
    Returns:
    --------
    torch.Tensor:
        Mel frequency spectrogram of size (…, n_mels, time).
    """
    audio_signal = torch.from_numpy(audio["array"])
    sample_rate = audio["sampling_rate"]
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        f_min=50,
        f_max=3500,
        n_mels=32,
    )
    mel_spec = mel_spectrogram(audio_signal)
    return mel_spec


def add_melspectrogram(example):
    example["input_ids"] = _compute_melspectrogram(example["audio"])
    return example


class DataCollatorxLSTM:
    def __init__(self):
        pass
