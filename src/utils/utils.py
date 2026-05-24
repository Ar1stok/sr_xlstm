"""Utility functions for ASR data processing and model debugging."""
import logging

import numpy as np
import os
import random
import zipfile
from typing import List, Tuple, Dict, Any
import torch
import torchaudio
import torchaudio.transforms
import io
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def make_length_filter(max_time: int = 512, max_labels: int = 256):
    """Create dataset filter function for time and label length limits.

    Filters samples where maximum dimension of mel-spectrogram exceeds max_time
    or label sequence exceeds max_labels. Handles both (time, n_mels) and
    (n_mels, time) input shapes.

    Args:
        max_time: Maximum allowed time steps in mel-spectrogram.
        max_labels: Maximum allowed label sequence length.

    Returns:
        Callable[[dict], bool]: Filter function for dataset filtering.
    """
    def _filter(example):
        mel = example['input_values']
        if not mel:
            return False

        # Handle both (time, n_mels) and (n_mels, time).
        n_rows = len(mel)
        n_cols = len(mel[0])
        time = max(n_rows, n_cols)

        return (time <= max_time and
                len(example['input_ids']) <= max_labels)

    return _filter


def compute_mel_stats(
        dataset,
        key: str = 'input_values',
        max_items: int | None = None
    ) -> tuple[float, float]:
    """Compute global mean and std across all mel-spectrograms.

    Normalizes shape to (time, n_mels) for consistent statistics. Uses running
    sum/sum-of-squares for numerical stability.

    Args:
        dataset: Dataset-like object indexable by integers.
        key: Dataset key containing mel-spectrogram arrays.
        max_items: Maximum number of items to process; None for all.

    Returns:
        Tuple[float, float]: Global (mean, std) across all mel elements.

    Raises:
        RuntimeError: If mel-spectrogram is not 2D.
    """
    sum_total = 0.0
    sum_sq_total = 0.0
    total_elements = 0

    n_items = len(dataset) if max_items is None else min(len(dataset),
                                                         max_items)

    for i in tqdm(range(n_items), desc='Compute mel statistics'):
        mel = np.array(dataset[i][key], dtype='float32')

        if mel.ndim != 2:
            raise RuntimeError(
                f'Invalid mel-spectrogram ndim={mel.ndim} at index {i}, '
                f'shape={mel.shape}')

        # Normalize to (time, n_mels).
        if mel.shape[0] < mel.shape[1]:
            mel = mel.T

        sum_total += mel.sum()
        sum_sq_total += (mel ** 2).sum()
        total_elements += mel.size

    mean = sum_total / total_elements
    variance = sum_sq_total / total_elements - mean ** 2
    std = np.sqrt(max(variance, 1e-8))

    logger.info('Mel statistics: mean=%.4f, std=%.4f (processed %d items)',
                mean, std, n_items)
    return float(mean), float(std)


def debug_model_forward(
    model,
    data_collator,
    raw_dataset,
    idxs: tuple[int, ...] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Debug model forward pass with sample batch.

    Runs specified samples through collator and model with debug mode enabled.
    Prints tensor shapes and lengths for validation.

    Args:
        model: PyTorch model with debug attribute and eval() support.
        data_collator: Data collator producing batch dict.
        raw_dataset: Raw dataset for sampling.
        idxs: Indices of samples to process.

    Returns:
        Model logits tensor from forward pass.

    Raises:
        KeyError: If required batch keys missing.
    """
    model.debug = True
    model.eval()

    batch = data_collator([raw_dataset[i] for i in idxs])

    with torch.no_grad():
        outputs = model(
            input_values=batch['input_values'],
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('input_ids'),
        )
        logits = outputs['logits']

    print(f'[DEBUG] Logits shape: {logits.shape}')
    print(f'[DEBUG] Input values shape: {batch["input_values"].shape}')
    if 'input_lengths' in batch:
        print(f'[DEBUG] Input lengths: {batch["input_lengths"]}')
    if 'targets_lengths' in batch:
        print(f'[DEBUG] Target lengths: {batch["targets_lengths"]}')

    model.debug = False
    return logits


def compute_mfcc_stats(
    zip_path: str,
    zip_list: List[str],
    n_samples_per_zip: int = 1000,
    sample_rate: int = 16000,
    mfcc_params: Dict[str, Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute global MFCC mean and std across n_samples_per_zip random samples from each zip.
    """
    mfcc_params = mfcc_params or {}
    mfcc_params.setdefault("sample_rate", sample_rate)
    mfcc_params.setdefault("n_mfcc", 13)
    if "melkwargs" not in mfcc_params:
        mfcc_params["melkwargs"] = {
            "n_fft": 400,
            "win_length": 400,
            "hop_length": 160,
            "f_min": 0.0,
            "f_max": 8000.0,
            "n_mels": 32,
            "center": False,
            "normalized": False,
            "mel_scale": "slaney",
        }
    transform = torchaudio.transforms.MFCC(**mfcc_params)

    all_mfccs = []

    for zip_name in zip_list:
        zip_path_full = os.path.join(zip_path, zip_name)
        if not os.path.isfile(zip_path_full):
            print(f"Skipping missing zip: {zip_path_full}")
            continue

        with zipfile.ZipFile(zip_path_full, "r") as zf:
            wav_files = [f for f in zf.namelist() if f.endswith(".wav")]
            selected = random.sample(wav_files, min(n_samples_per_zip, len(wav_files)))

            for wav_path in selected:
                try:
                    with zf.open(wav_path) as wav_file:
                        logger.info(f"Processing {wav_path}")
                        wav_bytes = wav_file.read()
                        wav_io = io.BytesIO(wav_bytes)
                        waveform, sr = torchaudio.load_with_torchcodec(wav_io, normalize=True)

                        std = waveform.std().item()
                        if std < 1e-8:
                            continue

                        mfcc = transform(waveform).squeeze(0)  # (n_mfcc, time)
                        all_mfccs.append(mfcc.flatten())

                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")

    if not all_mfccs:
        raise ValueError("No valid samples found for MFCC stat computation.")

    # (K, ) по всем фреймам и коэффициентам
    stacked = torch.cat(all_mfccs, dim=0)
    mean = stacked.mean().unsqueeze(0)  # (1,)
    std = stacked.std().unsqueeze(0) + 1e-6  # (1,)

    return mean, std