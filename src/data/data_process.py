""" Data Processing: mel-spectrograms, tokenization, and dynamic padding. """
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio.transforms as T
from datasets import Dataset, DatasetDict
from librosa.feature import melspectrogram
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def to_melspectrogram(
    dataset: Dataset | DatasetDict,
    method: str = "default"
) -> Dataset | DatasetDict:
    """Add melspectrogram to dataset as "input_mel_ids" column.
  
    Args:
        dataset: HuggingFace dataset with audiodata.
        method: Computation method ("default", "librosa" or "torch").
    
    Returns:
        Dataset or DatasetDict with an added "input_mel_ids" column.
    
    Raises:
        ValueError: If method is not in the supported values.
    """
    logger.info("Starting mel-spectrogram computation with method: %s", method)
    def _process_librosa(example):
        """Compute melspectrogram using librosa"""
        audio = example["audio"]
        audio_signal = audio["array"]
        sampling_rate = audio["sampling_rate"]

        mel_spec = melspectrogram(
            y=audio_signal,
            sr=sampling_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            fmin=50,
            fmax=3500,
            n_mels=32,
        )
        example["input_mel_ids"] = torch.from_numpy(mel_spec)
        return example

    def _process_torch(example):
        """Compute melspectrogram using torchaudio"""
        audio = example["audio"]
        audio_signal = torch.from_numpy(audio["array"])
        sampling_rate = audio["sampling_rate"]
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=50,
            f_max=3500,
            n_mels=32,
        )
        mel_spec = mel_spectrogram(audio_signal)
        example["input_mel_ids"] = mel_spec
        return example

    if method in ("default", "librosa"):
        processed_data = dataset.map(
            _process_librosa,
            batched=False,
            desc="Computing mel-spectrograms (librosa)"
        )
    elif method == "torch":
        processed_data = dataset.map(
            _process_torch,
            batched=False,
            desc="Computing mel-spectrograms (torchaudio)"
        )
    else:
        error_msg = f"Unknown method: {method}. Supported: 'torch', 'librosa'"
        raise ValueError(error_msg)
    logger.info("Mel-spectrogram computation completed")
    return processed_data


def tokenize_labels(
    tokenizer_path: str,
    dataset: DatasetDict | Dataset
) -> Dataset | DatasetDict:
    """Tokenizes transcriptions and adds "input_label_ids" column.
  
    Args:
        tokenizer_path: Path or model id for the tokenizer
            (Whisper tokenizer is recommended for ASR).
        dataset: HuggingFace Dataset with a "transcription" column.
    
    Returns:
        Dataset or DatasetDict with an added "input_label_ids" column.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info("Tokenizer loaded (vocab_size=%d)", tokenizer.vocab_size)

    def _tokenize_function(example):
        """Tokenizes a single example."""
        tokenized = tokenizer(
            example["transcription"],
            truncation=False,
            return_tensors=None,
        )
        example["input_label_ids"] = tokenized["input_ids"]
        return example
    tokenized_data = dataset.map(
        _tokenize_function,
        batched=False,
        desc="Tokenizing transcriptions",
    )
    logger.info("Tokenize complete")
    return tokenized_data


@dataclass
class DataCollatorASRWithPadding:
    """
    Data collator for ASR tasks with xLSTM models.
    
    Dynamically pads:
    - input_mel_ids: 2D mel-spectrogram sequences (n_mels, time_steps)
    - input_label_ids: 1D token sequences
    
    Attributes:
        max_input_length (int, optional): Maximum mel-spectrogram time dim.
            If None, uses the longest sequence in the batch.
        max_labels_length (int, optional): Maximum label sequence length.
            If None, uses the longest sequence in the batch.
        padding_value (float): Value used for padding mel-spectrograms.
        labels_pad_token_id (int): Token ID used for padding labels.
    """
    max_input_length: Optional[int] = None
    max_labels_length: Optional[int] = None
    padding_value: float = 0.0
    labels_pad_token_id: int = -100
    def __call__(
        self, batch: List[Dict[str, Union[List, np.ndarray, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of examples for ASR training.
        
        Args:
            batch: List of dictionaries with keys:
                - 'input_mel_ids': 2D array of shape (n_mels, time_steps)
                - 'input_label_ids': 1D array of token IDs
                - 'audio' (optional): Audio metadata (discarded)
                - 'transcription' (optional): Original text (discarded)
        
        Returns:
            Dict with batched tensors:
                - 'input_ids': Padded mel-spectrograms
                - 'labels': Padded token sequences
                - 'input_ids_lengths': Actual lengths before padding
                - 'labels_lengths': Actual label lengths before padding
        """
        if not batch:
            raise RuntimeError("Batch is empty!")
        # Extract mel-spectrograms and labels
        input_ids_list = []
        labels_list = []
        input_ids_lengths = []
        labels_lengths = []
        for example in batch:
            # Validate required keys
            if "input_mel_ids" not in example:
                raise RuntimeError(
                    f"Example missing 'input_mel_ids'. "
                    f"Available keys: {list(example.keys())}"
                )
            if "input_label_ids" not in example:
                raise RuntimeError(
                    f"Example missing 'input_label_ids'. "
                    f"Available keys: {list(example.keys())}"
                )
            # Convert to numpy with explicit dtype
            mel_ids = np.array(example["input_mel_ids"], dtype="float32")
            label_ids = np.array(example["input_label_ids"], dtype="int64")
            # === FORM NORMALIZATION ===
            # Converting mel_ids to 2D (n_mels, time)
            if mel_ids.ndim != 2:
                raise RuntimeError(
                    f"mel_ids should be 2D after normalization, "
                    f"got {mel_ids.ndim}D "
                    f"shape {mel_ids.shape}"
                )
            n_mels_ex, time_ex = mel_ids.shape
            # Check the axes:
            # if suddenly (time, n_mels) instead of (n_mels, time)
            # We assume that n_mels always == 32, so if the first axis >> 32,
            # it is probably time
            if n_mels_ex > 64 and time_ex <= 64:
                # Maybe (time, 32) → transpose (32, time)
                mel_ids = mel_ids.T
                n_mels_ex, time_ex = mel_ids.shape
            # Save the original length over time (before padding/truncate)
            input_ids_lengths.append(time_ex)
            # Check label_ids
            if label_ids.ndim != 1:
                raise RuntimeError(
                    f"label_ids should be 1D, "
                    f"got {label_ids.ndim}D shape {label_ids.shape}"
                )
            labels_lengths.append(len(label_ids))
            # Add to the list for further processing
            input_ids_list.append(mel_ids)
            labels_list.append(label_ids)
        # === CALCULATE THE MAXIMUM DIMENSIONS ===
        max_n_mels = max(ids.shape[0] for ids in input_ids_list)
        max_time_in_batch = max(ids.shape[1] for ids in input_ids_list)
        if self.max_input_length is not None:
            max_time_steps = min(max_time_in_batch, self.max_input_length)
        else:
            max_time_steps = max_time_in_batch
        if self.max_labels_length is not None:
            max_label_len = self.max_labels_length
        else:
            max_label_len = max(len(labels) for labels in labels_list)
        # === PADDING MEL-SPECTROGRAMS ===
        padded_input_ids = torch.full(
            (len(batch), max_n_mels, max_time_steps),
            fill_value=self.padding_value,
            dtype=torch.float32
        )
        for i, mel_ids in enumerate(input_ids_list):
            mel_ids = mel_ids.astype("float32")  # guarantees float32
            n_mels_ex, time_ex = mel_ids.shape
            # (1) Time truncation > (if needed)
            if time_ex > max_time_steps:
                mel_ids = mel_ids[:, :max_time_steps]
                time_ex = max_time_steps
            # (2) Padding by time < (if needed)
            if time_ex < max_time_steps:
                pad_time = max_time_steps - time_ex
                mel_ids = np.pad(
                    mel_ids,
                    ((0, 0), (0, pad_time)),
                    mode="constant",
                    constant_values=self.padding_value
                )
                time_ex = max_time_steps
            # (3) Check before assignment
            expected_shape = (max_n_mels, max_time_steps)
            if mel_ids.shape != expected_shape:
                raise RuntimeError(
                    f"Bad mel shape after padding: {mel_ids.shape}, "
                    f"expected {expected_shape}"
                )
            padded_input_ids[i] = torch.from_numpy(mel_ids)
        # === PADDING LABELS ===
        padded_labels = torch.full(
            (len(batch), max_label_len),
            fill_value=self.labels_pad_token_id,
            dtype=torch.int64
        )
        for i, labels in enumerate(labels_list):
            labels = labels.astype("int64")
            # truncation if needed
            if len(labels) > max_label_len:
                labels = labels[:max_label_len]
            padded_labels[i, :len(labels)] = torch.from_numpy(labels)
        # === FINAL BATCH ===
        batch_dict = {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "input_ids_lengths": torch.tensor(
                input_ids_lengths,
                dtype=torch.int32
            ),
            "labels_lengths": torch.tensor(
                labels_lengths,
                dtype=torch.int32
            ),
        }
        return batch_dict
