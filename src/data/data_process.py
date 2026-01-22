"""Data process for ASR: mel-spectrogram, tokenize, and dynamic padd."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio.transforms as T
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, Wav2Vec2Processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def to_waveform(
    dataset: Dataset | DatasetDict,
    processor: Wav2Vec2Processor,
) -> Dataset | DatasetDict:
    """Convert raw audio data to normalized waveform tensors for Wav2Vec2.

    This function processes datasets containing 'audio' column and 
    adds 'input_values' column with normalized float32 waveforms 
    suitable for Wav2Vec2 models.

    Args:
        dataset: Input dataset(s) containing 'audio' column with raw audio data.
        processor: Pre-trained Wav2Vec2 processor for audio normalization and
        resampling.

    Returns:
        Processed dataset with additional 'input_values' column containing
        normalized torch.Tensor waveforms (float32, range [-1.0, 1.0]).
    """
    def _process_batched(example):
        audio_array = example["audio"]["array"]
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False
        )
        example["input_values"] = inputs.input_values[0]
        return example

    logger.info("Starting waveform processing...")
    processed = dataset.map(
        _process_batched,
        batched=False,
        remove_columns=None)
    logger.info("Processing completed, added 'input_values' column")

    return processed


def to_melspectrogram(
    dataset: Dataset | DatasetDict,
    n_mels: int = 32,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    batch_size: int = 64,
) -> Dataset | DatasetDict:
    """Compute batched mel-spectrograms with normalization.

    Uses PyTorch transforms for CPU/GPU compatibility. 
    Outputs (T, n_mels) shape.
    Removes 'audio' column after processing.
    
    Args:
        dataset: Dataset with 'audio' column (dict: array, sampling_rate).
        n_mels: Number of Mel frequency bins.
        sr: Audio sampling rate in Hz.
        n_fft: FFT window length.
        hop_length: Hop length between frames.
        batch_size: Batch size for processing.
    
    Returns:
        Dataset/DatasetDict with 'input_values' column 
        of shape (time_steps, n_mels).
    """
    logger.info(
        "PyTorch batched mel (n_mels=%d, batch_size=%d)", 
        n_mels,
        batch_size)

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=sr / 2,
        n_mels=n_mels,
        normalized=False,
        mel_scale="slaney")

    def _process_batched(batched):
        mel_list = []
        for a in batched["audio"]:
            waveform = torch.from_numpy(
                np.array(a["array"], dtype=np.float32)).unsqueeze(0)  # (1, L)
            mel = mel_transform(waveform).squeeze(0)  # (F, T)
            mel = T.AmplitudeToDB(stype="power")(mel)
            # Normalize per spec: subtract mean, divide by std along time.
            mean = mel.mean(dim=-1, keepdim=True)
            std = mel.std(dim=-1, keepdim=True) + 1e-6
            mel_norm = (mel - mean) / std
            mel_list.append(mel_norm.t().numpy())  # (T, F)
        batched["input_values"] = mel_list
        return batched

    processed = dataset.map(
        _process_batched,
        batched=True,
        batch_size=batch_size,
        desc="PyTorch MelSpectrogram",
        remove_columns=["audio"],
    )

    logger.info("Mel-spectrogram computation complete.")
    return processed


def tokenize_labels(
    tokenizer_path: str,
    dataset: DatasetDict | Dataset
) -> Dataset | DatasetDict:
    """Tokenizes transcriptions and adds 'input_ids' column.
    
    Args:
        tokenizer_path: Path to pretrained tokenizer directory.
        dataset: Dataset with a "transcription" column.
    
    Returns:
        Dataset/DatasetDict with added 'input_ids' column (list of int).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(
        "Tokenizer loaded (vocab_size=%d, blank_id=%d)", 
        len(tokenizer),
        tokenizer.pad_token_id)

    def _tokenize_function(example):
        model_inputs = tokenizer(
            example["transcription"],
            return_tensors=None,
            add_special_tokens=False
        )
        label_ids = np.array(model_inputs["input_ids"], dtype=np.int64)
        if tokenizer.pad_token_id is not None:
            pad_id = np.int64(tokenizer.pad_token_id)
            label_ids[label_ids == pad_id] = np.int64(-100)
        example["input_ids"] = label_ids.tolist()
        return example

    tokenized_data = dataset.map(
        _tokenize_function,
        batched=False,
        desc="Tokenizing transcriptions",
        remove_columns=[]
    )
    logger.info("Tokenization complete")
    return tokenized_data


@dataclass
class DataCollatorASRWithPadding:
    """Data collator for ASR with mel-spectrograms and dynamic padding.

    Pads input_values to (batch, time, n_mels) and labels to (batch, seq_len).
    Supports fixed n_mels, attention_mask, length tracking, and
    padding to multiple. Handles transposition detection for input shapes.
    """
    fixed_n_mels: Optional[int] = 32
    max_input_length: Optional[int] = None
    max_labels_length: Optional[int] = None
    padding_value: float = 0.0
    labels_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self,
        batch: List[Dict[str, Union[List, np.ndarray, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Pad batch dynamically.

        Args:
        batch: List of dicts with 'input_values' (2D mel np.array) 
          and 'input_ids'.

        Returns:
        Dict with padded tensors: input_values, input_ids (labels),
          attention_mask, input_lengths, targets_lengths.

        Raises:
        RuntimeError: If batch empty or missing keys/shapes invalid.
        """
        if not batch:
            raise RuntimeError("Batch is empty!")

        input_values_list = []
        labels_list = []
        input_lengths = []
        targets_lengths = []

        # Extract and validate
        for example in batch:
            if "input_values" not in example:
                raise RuntimeError(f"Example missing 'input_values'. Keys: \
                                   {list(example.keys())}")
            if "input_ids" not in example:
                raise RuntimeError(f"Example missing 'input_ids'. Keys: \
                                   {list(example.keys())}")

            mel_spec = np.array(example["input_values"], dtype="float32")
            label_ids = np.array(example["input_ids"], dtype="int64")

            if mel_spec.ndim != 2:
                raise RuntimeError(f"input_values must be 2D, \
                                   got {mel_spec.ndim}D {mel_spec.shape}")

            # Auto-transpose if needed: ensure (n_mels, time).
            n_mels_ex, time_ex = mel_spec.shape
            if time_ex < n_mels_ex:
                mel_spec = mel_spec.T
                n_mels_ex, time_ex = time_ex, n_mels_ex

            if self.fixed_n_mels and n_mels_ex != self.fixed_n_mels:
                raise RuntimeError(f"Expected n_mels={self.fixed_n_mels}, \
                                   got {n_mels_ex}")

            input_lengths.append(time_ex)
            targets_lengths.append(len(label_ids))
            input_values_list.append(mel_spec)
            labels_list.append(label_ids)

        batch_size = len(batch)
        n_mels = self.fixed_n_mels or max(spec.shape[0]
                                          for spec in input_values_list)

        # Compute max lengths with padding multiples.
        max_time_in_batch = max(spec.shape[1] for spec in input_values_list)
        max_time_steps = self._pad_to_multiple(
            min(max_time_in_batch, self.max_input_length or float("inf")))
        max_label_len = self._pad_to_multiple(
            min(max(len(lab) for lab in labels_list),
                self.max_labels_length or float("inf")))

        # Pad input_values: (batch, n_mels, max_time) → transpose to (B, T, C)
        padded_input_values = torch.full(
            (batch_size, n_mels, max_time_steps),
            self.padding_value,
            dtype=torch.float32,
        )
        attention_mask = torch.zeros(
            (batch_size, max_time_steps),
            dtype=torch.long)

        for i, mel_spec in enumerate(input_values_list):
            mel_spec = mel_spec.astype("float32")
            n_mels_ex, time_ex = mel_spec.shape

            # Truncate
            if time_ex > max_time_steps:
                mel_spec = mel_spec[:, :max_time_steps]
                time_ex = max_time_steps

            # Pad time dim
            if time_ex < max_time_steps:
                pad_time = max_time_steps - time_ex
                mel_spec = np.pad(
                    mel_spec,
                    ((0, 0), (0, pad_time)),
                    mode="constant",
                    constant_values=self.padding_value)

            # Pad n_mels (rare)
            if n_mels_ex < n_mels:
                pad_mels = n_mels - n_mels_ex
                mel_spec = np.pad(
                    mel_spec,
                    ((0, pad_mels), (0, 0)),
                    mode="constant",
                    constant_values=self.padding_value)

            padded_input_values[i] = torch.from_numpy(mel_spec)
            attention_mask[i, :time_ex] = 1

        padded_input_values = padded_input_values.transpose(1, 2) # (B, T, C)

        # Pad labels with -100 masking.
        padded_labels = torch.full(
            (batch_size, max_label_len),
            self.labels_pad_token_id,
            dtype=torch.long)
        label_mask = torch.zeros((batch_size, max_label_len), dtype=torch.long)

        for i, labels in enumerate(labels_list):
            labels = labels.astype("long")
            trunc_len = min(len(labels), max_label_len)
            padded_labels[i, :trunc_len] = torch.from_numpy(labels[:trunc_len])
            label_mask[i, :trunc_len] = 1

        # Apply masking to labels (like CTCWithPadding)
        padded_labels = padded_labels.masked_fill(
            label_mask.eq(0),
            self.labels_pad_token_id)

        return {
            "input_values": padded_input_values,
            "input_ids": padded_labels,
            "attention_mask": attention_mask,
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
            "targets_lengths": torch.tensor(targets_lengths, dtype=torch.long),
        }

    def _pad_to_multiple(
            self,
            length: int,
            multiple: Optional[int] = None
        ) -> int:
        """Pad length to nearest multiple for efficient conv/TensorCores.

        Args:
        length: Original length.
        multiple: Padding multiple; defaults to self.pad_to_multiple_of.

        Returns:
        Padded length.
        """
        if multiple is None:
            multiple = getattr(self, "pad_to_multiple_of", None) or 1
        if multiple > 1:
            length = ((length + multiple - 1) // multiple) * multiple
        return length


@dataclass
class DataCollatorCTCWithPadding:
    """Standard CTC data collator using Wav2Vec2Processor.

    Dynamically pads waveforms and labels. Replaces padding in labels with -100.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(
            self,
            processor,
            padding=True,
            max_length=None,
            max_length_labels=None,
            sample_rate=16_000
        ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.sample_rate = sample_rate

    def __call__(
            self,
            features: List[Dict[str, Union[List[int], torch.Tensor]]]
            ) -> Dict[str, torch.Tensor]:
        """Pad batch using processor.

        Args:
        features: List of feature dicts with 'input_values' and 'input_ids'.

        Returns:
        Padded batch dict.
        """
        # Split for different padding strategies.
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        print(f"input_features: {input_features}")
        label_features = [{"input_ids": feature["input_ids"]}
                          for feature in features]
        print(f"label_features: {label_features}")

        # Pad audio features.
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels.
        input_ids = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Mask padding for loss.
        input_ids = input_ids["input_ids"].masked_fill(
            input_ids.attention_mask.ne(1), -100)

        batch["input_ids"] = input_ids
        return batch
