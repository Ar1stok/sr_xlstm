"""Utility functions for ASR data processing and model debugging."""
import logging

import numpy as np
import torch
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
