import glob
import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset
from torchcodec.decoders import AudioDecoder

logger = logging.getLogger(__name__)

def get_duration_sec(wav_bytes: bytes) -> float:
    decoder = AudioDecoder(io.BytesIO(wav_bytes))
    md = decoder.metadata
    return md.duration_seconds_from_header

def iter_zip_items(
    zip_path: Path,
    max_audio_seconds: float,
) -> Iterator[Dict]:
    """
    Итерируется по одному zip и даёт примеры:
    {
        "audio": {"array": np.ndarray, "sampling_rate": int},
        "transcription": str,
    }
    """
    logger.info("Opening zip: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        wav_files = [m for m in members if m.lower().endswith(".wav")]
        wav_files.sort()

        logger.info("Found %d wav files in %s", len(wav_files), zip_path)

        for wav_name in wav_files:
            base = os.path.splitext(wav_name)[0]
            txt_name = base + ".txt"
            if txt_name not in members:
                continue

            # читаем транскрипт
            with zf.open(txt_name, "r") as f_txt:
                transcription = f_txt.read().decode("utf-8").strip()

            # читаем и декодируем wav
            with zf.open(wav_name, "r") as f_wav:
                raw = f_wav.read()

            # проверка на длительность аудио
            duration_sec = get_duration_sec(raw)
            if duration_sec > max_audio_seconds:
                continue

            audio_io = io.BytesIO(raw)
            waveform, sr = sf.read(audio_io, dtype="float32")

            yield {
                "audio": {
                    "array": waveform,
                    "sampling_rate": sr,
                },
                "transcription": transcription,
            }


def build_chunks_from_zip(
    zip_path: Path,
    sampling_rate: int = 16000,
    max_examples_per_chunk: int = 50000,
    max_audio_seconds: float = 15.0,
) -> List[Dataset]:
    """
    Разбивает содержимое zip на чанки по max_examples_per_chunk и
    для каждого чанка создаёт отдельный Dataset с Audio-фичей.
    """
    features = Features({
        "audio": Audio(sampling_rate=sampling_rate),
        "transcription": Value("string"),
    })

    datasets = []
    buffer = []
    total = 0
    chunk_idx = 0

    for example in iter_zip_items(zip_path, max_audio_seconds):
        buffer.append(example)
        total += 1

        if len(buffer) >= max_examples_per_chunk:
            logger.info(
                "Building chunk %d from %s with %d examples (total so far: %d)",
                chunk_idx,
                zip_path,
                len(buffer),
                total,
            )
            ds_chunk = Dataset.from_list(buffer, features=features)
            datasets.append(ds_chunk)

            buffer = []
            chunk_idx += 1

    # остаток
    if buffer:
        logger.info(
            "Building last chunk %d from %s with %d examples (total: %d)",
            chunk_idx,
            zip_path,
            len(buffer),
            total,
        )
        ds_chunk = Dataset.from_list(buffer, features=features)
        datasets.append(ds_chunk)

    logger.info(
        "Finished building %d chunks from %s, total examples: %d",
        len(datasets),
        zip_path,
        total,
    )
    return datasets

def zip_to_parquet_chunks(
    root_path: str,
    save_root: str,
    sampling_rate: int = 16000,
    max_examples_per_chunk: int = None,
    max_audio_seconds: float = 15.0,
) -> None:
    """
    Для каждого part_*.zip создаёт несколько parquet:
    part_0.zip -> {save_root}/part_0/part_0_chunk0.parquet, chunk1.parquet, ...
    """
    os.makedirs(save_root, exist_ok=True)
    root = Path(root_path)

    for zip_path in sorted(root.glob("part_37.zip")):
        logger.info("Processing zip: %s", zip_path)

        ds_chunks = build_chunks_from_zip(
            zip_path,
            sampling_rate=sampling_rate,
            max_examples_per_chunk=max_examples_per_chunk,
            max_audio_seconds=max_audio_seconds,
        )

        # out_dir = os.path.join(save_root, zip_path.stem)
        os.makedirs(save_root, exist_ok=True)

        for i, ds_chunk in enumerate(ds_chunks):
            out_file = os.path.join(
                save_root,
                f"{zip_path.stem}_chunk{i}.parquet",
            )
            logger.info("Saving chunk %d to %s", i, out_file)
            ds_chunk.to_parquet(out_file)
            del ds_chunk

        logger.info("Finished %s", zip_path)


def load_from_parquet_parts(path: str, splits: list[str], split_name: str = "train"):
    data_files = {split: os.path.join(path, f'{split}.parquet')
                  for split in splits}
    # Загружаем всё в один split
    dataset = load_dataset('parquet',
                           data_files=list(data_files.values()),
                           split=split_name)
    return dataset