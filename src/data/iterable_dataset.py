import io
import logging
import os
from typing import List, Optional, Dict, Any
import zipfile
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchcodec.decoders import AudioDecoder
import torchaudio
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class ZipAudioDataset(IterableDataset):
    def __init__(
        self,
        zip_path: str,
        zip_list: List[str],
        chunk_size: int = 1000,
        tokenizer_path: str = 'src/utils/tokenizer',
        mfcc_params: Optional[Dict[str, Any]] = None,
       
        # New Limits
        max_audio_seconds: Optional[float] = None,
        max_tokens: Optional[int] = None,
        sample_rate: int = 16000,
        debug: bool = False,
    ):
        logger.info("Init started")
        if not os.path.isdir(zip_path):
            raise ValueError(f"Directory does not exist: {zip_path}")


        self.zip_path = zip_path
        self.zip_list = [f for f in zip_list if f.endswith('.zip')]
        if not self.zip_list:
            raise ValueError("The list of zip files is empty or does not contain .zip")

        missing = [f for f in self.zip_list if not os.path.isfile(os.path.join(zip_path, f))]
        if missing:
            raise FileNotFoundError(f"Missing zip files: {missing}")
        logger.info("Path find")

        self.chunk_size = chunk_size
        self.current_zip_idx = 0
        self.current_file_idx = 0
        self.buff_counter = 0
        self.buffer = []
        logger.info("Variable init")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("Tokenizer init")

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
                "mel_scale": "slaney"
            }
        self.transform = torchaudio.transforms.MFCC(**mfcc_params)

        # Limits and debug mode
        self.max_audio_seconds = max_audio_seconds
        self.max_tokens = max_tokens
        self.sample_rate = sample_rate
        self.debug = debug
        logger.info("Init done")
   
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"Worker_Info: {worker_info}")
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        logger.info(
            f"Worker init → worker_id={worker_id}, num_workers={num_workers}"
        )

        yield from self._fill_buffer(worker_id, num_workers)


    def _get_duration_sec(self, wav_bytes: bytes) -> float:
        decoder = AudioDecoder(io.BytesIO(wav_bytes))
        md = decoder.metadata
        return md.duration_seconds_from_header


    def _mfcc_transform(self, wav_bytes):
        wav_io = io.BytesIO(wav_bytes)
        waveform, sr = torchaudio.load_with_torchcodec(
            wav_io,
            normalize=True
        )

        std = waveform.std().item()
        if std < 1e-8:
            return None
        
        mfcc = self.transform(waveform).squeeze(0) # (n_mfcc, time)
        return mfcc


    def _tokenize_text(self, text: str) -> List[int]:
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors=None
        )
        label_ids = np.array(encoding["input_ids"], dtype=np.int64)
        if self.tokenizer.pad_token_id is not None:
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return label_ids.tolist()


    def _fill_buffer(self, worker_id: int, num_workers: int):
        current_zip_idx = 0


        while current_zip_idx < len(self.zip_list):
            zip_name = self.zip_list[current_zip_idx]
            zip_path = os.path.join(self.zip_path, zip_name)


            with zipfile.ZipFile(zip_path, "r") as zf:
                file_list = sorted(zf.namelist())
                file_set = set(file_list)

                wav_files = [f for f in file_list if f.endswith(".wav")]

                # idx % num_workers == worker_id
                for local_idx, wav_path in enumerate(wav_files):
                    if num_workers > 1 and (local_idx % num_workers != worker_id):
                        continue

                    txt_path = wav_path.replace(".wav", ".txt")
                    if txt_path not in file_set:
                        continue

                    try:
                        with zf.open(wav_path) as wav_file:
                            wav_bytes = wav_file.read()

                        duration_sec = self._get_duration_sec(wav_bytes)
                        if self.max_audio_seconds and duration_sec > self.max_audio_seconds:
                            if self.debug:
                                logger.info(f"Skip long audio {duration_sec:.2f}s: {wav_path}")
                            continue

                        input_values = self._mfcc_transform(wav_bytes)
                        if input_values is None:
                            continue

                        with zf.open(txt_path) as txt_file:
                            text = txt_file.read().decode("utf-8").strip()
                        if len(text) < 3:
                            continue

                        input_ids = self._tokenize_text(text)
                        if self.max_tokens and len(input_ids) > self.max_tokens:
                            continue

                        yield {
                            "input_values": input_values,
                            "input_ids": input_ids,
                            "transcription": text,
                        }

                    except Exception as e:
                        logger.warning(f"Processing error {wav_path}: {e}")

            current_zip_idx += 1
           
            logger.info(
                f"Worker {worker_id}/{num_workers}: complete pass over {zip_name}"
            )

        logger.info(
            f"Worker {worker_id}/{num_workers}: completed one full pass over all zip files"
        )