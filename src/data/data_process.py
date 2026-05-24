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
            mel_list.append(mel_norm.numpy())  # (F, T)
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


def to_mfcc(
    dataset: Dataset | DatasetDict,
    n_mfcc: int = 13,
    n_mels: int = 32,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    batch_size: int = 64,
    device: str = None,  # 'cuda' или None (cpu)
) -> Dataset | DatasetDict:
    """Compute batched MFCC with DB-scaling and normalization (mean/std per spec).

    Uses PyTorch transforms for CPU/GPU. Outputs (T, n_mfcc) shape, consistent with mel.
    Removes 'audio' column. Assumes dataset.cast_column("audio", Audio(sampling_rate=sr)).

    Args:
        dataset: Dataset with 'audio' column (dict: array, sampling_rate).
        n_mfcc: Number of MFCC coefficients.
        n_mels: Mel bins for internal MelSpectrogram.
        sr: Target sampling rate.
        n_fft, hop_length: STFT params.
        batch_size: Batch size.
        device: 'cuda' if available for speedup.

    Returns:
        Dataset with 'input_values' (time_steps, n_mfcc).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("PyTorch batched MFCC (n_mfcc=%d, n_mels=%d, device=%s, batch=%d)", 
                n_mfcc, n_mels, device, batch_size)

    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "win_length": n_fft,
            "hop_length": hop_length,
            "f_min": 0.0,
            "f_max": sr / 2,
            "n_mels": n_mels,
            "center": False,
            "normalized": False,
            "mel_scale": "slaney"
        },
    ).to(device)

    def _process_batched(batched):
        mfcc_list = []
        for a in batched["audio"]:
            waveform = torch.from_numpy(np.array(a["array"], dtype=np.float32)).unsqueeze(0).to(device)  # (1, L)
            mfcc = mfcc_transform(waveform).squeeze(0)  # (n_mfcc, T)
            mfcc_list.append(mfcc.cpu().numpy())  # (n_mfcc, T)
        batched["input_values"] = mfcc_list
        return batched

    processed = dataset.map(
        _process_batched,
        batched=True,
        batch_size=batch_size,
        desc="PyTorch MFCC",
        remove_columns=["audio"],
    )

    logger.info("MFCC computation complete.")
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
        label_ids = label_ids.tolist()
        if len(label_ids) > 512:
            logger.info(f"Number of tokens more than 512: {len(label_ids)}")
        example["input_ids"] = label_ids
        return example

    tokenized_data = dataset.map(
        _tokenize_function,
        batched=False,
        desc="Tokenizing transcriptions",
        remove_columns=[]
    )
    logger.info("Tokenization complete")
    return tokenized_data

# TODO: Изменить размерность на (B, max_time, n_features) и 
# сделать требование одинакового формата
@dataclass
class DataCollatorASRWithPadding:
    fixed_n_mels: Optional[int] = 32
    max_input_length: Optional[int] = None
    max_labels_length: Optional[int] = None
    padding_value: float = 0.0
    labels_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = 8           # для conv / tensor cores
    pad_to_multiple_of_labels: Optional[int] = 8

    def __call__(self, batch: List[Dict]):
        if not batch:
            raise RuntimeError("Batch is empty!")

        input_values_list = []
        labels_list = []
        input_lengths = []
        targets_lengths = []

        for example in batch:
            mel_spec = np.array(example["input_values"], dtype="float32")
            label_ids = np.array(example["input_ids"], dtype="int64")

            if mel_spec.ndim != 2 or mel_spec.shape[0] != self.fixed_n_mels:
                raise ValueError(f"Неверный mel: {mel_spec.shape}")

            time_ex = mel_spec.shape[1]
            input_values_list.append(mel_spec)
            input_lengths.append(time_ex)
            labels_list.append(label_ids)
            targets_lengths.append(len(label_ids))

        # Вычисляем максимумы (без жёсткой обрезки — датасет уже отфильтровал)
        max_time_steps = max(input_lengths)
        if self.pad_to_multiple_of:
            max_time_steps = ((max_time_steps + self.pad_to_multiple_of - 1) 
                            // self.pad_to_multiple_of * self.pad_to_multiple_of)

        max_label_len = max(targets_lengths)
        if self.pad_to_multiple_of_labels:
            max_label_len = ((max_label_len + self.pad_to_multiple_of_labels - 1) 
                           // self.pad_to_multiple_of_labels * self.pad_to_multiple_of_labels)

        # Паддинг аудио
        padded_input_values = torch.full(
            (len(batch), max_time_steps, self.fixed_n_mels),
            self.padding_value, dtype=torch.float32
        )
        attention_mask = torch.zeros((len(batch), max_time_steps), dtype=torch.long)

        for i, mel in enumerate(input_values_list):
            t = min(len(mel[0]), max_time_steps)  # на всякий случай
            padded_input_values[i, :t] = torch.from_numpy(mel[:, :t].T)  # (time, n_mels)
            attention_mask[i, :t] = 1

        # Паддинг меток
        padded_labels = torch.full(
            (len(batch), max_label_len),
            self.labels_pad_token_id, dtype=torch.long
        )
        label_mask = torch.zeros((len(batch), max_label_len), dtype=torch.long)

        for i, lbl in enumerate(labels_list):
            seq_len = len(lbl)
            padded_labels[i, :seq_len] = torch.from_numpy(lbl.astype("long"))
            label_mask[i, :seq_len] = 1

        padded_labels = padded_labels.masked_fill(label_mask.eq(0), self.labels_pad_token_id)

        return {
            "input_values": padded_input_values,     # (B, T, C)
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
        label_features = [{"input_ids": feature["input_ids"]}
                          for feature in features]

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
