"""Tokenizer creation utilities for CTC-based ASR datasets."""
import json
import logging
import os
from typing import Optional

from datasets import DatasetDict
from tqdm import tqdm
from transformers import Wav2Vec2CTCTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_corpus_file(
        dataset: DatasetDict,
        output_file: Optional[str] = 'dataset_corpus.txt'
    ) -> None:
    """Extract all transcriptions to single corpus file for tokenizer training.

    Writes clean transcriptions from train/valid/test splits.
    Skips empty transcriptions.

    Args:
        dataset: DatasetDict with 'transcription' column 
          in train/valid/test splits.
        output_file: Output path for corpus file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        splits = (
            ['train', 'valid', 'test']
            if 'valid' in dataset
            else ['train', 'test']
        )
        for split in splits:
            for item in tqdm(dataset[split], desc=f'Processing {split}'):
                text = item.get('transcription', '').strip()
                if text:
                    # Replace newlines with spaces, normalize whitespace.
                    clean_text = ' '.join(text.replace('\n', ' ').split())
                    if clean_text:
                        f.write(clean_text + '\n')

    print(f'corpus file created: {output_file}')


def setup_char_tokenizer(corpus_file: str, save_path: str):
    """Create character-level CTC tokenizer from corpus file.

    Extracts unique characters, creates vocab with special tokens (<pad>,
    <unk>, <blank>, |), saves vocab.json and tokenizer files.

    Args:
        corpus_file: Path to corpus text file.
        save_path: Directory to save tokenizer files.

    Returns:
        Initialized Wav2Vec2CTCTokenizer instance.

    Raises:
        FileNotFoundError: If corpus_file does not exist.
        OSError: If save_path cannot be created.
    """
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f'Corpus file not found: {corpus_file}')

    # Read and extract unique characters.
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_text = f.read()

    # Sorted unique characters + CTC special tokens.
    special_tokens = ['<pad>', '<unk>', '<blank>', '|']
    chars = sorted(set(''.join(corpus_text.split())))
    vocab = {char: idx for idx, char in enumerate(special_tokens + chars)}

    # Ensure save directory exists.
    os.makedirs(save_path, exist_ok=True)
    vocab_path = os.path.join(save_path, 'vocab.json')

    # Save vocab.json for Wav2Vec2CTCTokenizer.
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Create and save tokenizer.
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        pad_token='<pad>',
        unk_token='<unk>',
        word_delimiter_token='|',
    )
    tokenizer.save_pretrained(save_path)

    logger.info('Character tokenizer saved to %s (vocab_size=%d)',
                save_path, len(vocab))
    return tokenizer
