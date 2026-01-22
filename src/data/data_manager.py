"""Dataset Manager: file loading, processing, and dataset creation."""
import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_files(root: str) -> List[Dict[str, str]]:
    """Load audio paths and transcriptions from nested directory structure.

    Expects subdirectories with paired .wav audio and .txt transcription files.

    Args:
        root: Root directory with audio/transcription subdirectories.

    Returns:
        List of dicts: [{'audio': str, 'transcription': str}, ...].
    """
    data = []
    root_path = Path(root)
    for path_fst in root_path.iterdir():
        if not path_fst.is_dir():
            continue
        for path_snd in path_fst.iterdir():
            if not path_snd.is_dir():
                continue
            for file_path in path_snd.glob('*.wav'):
                txt_path = file_path.with_suffix('.txt')
                if not txt_path.exists():
                    continue
                transcription = txt_path.read_text(encoding='utf8').strip()
                data.append({
                        'audio': str(file_path),
                        'transcription': transcription
                    })
    return data


def split_data(dataset: Dataset, split_config: Dict[str, Any]) -> DatasetDict:
    """Split HuggingFace Dataset into train, validation, and/or test sets.

    Supports flexible 2-way or 3-way splits with proportions or absolute sizes.
    Ensures reproducibility via seed and validates configuration.

    Args:
        dataset: Input HuggingFace Dataset to split.
        split_config: Dict with split parameters:
            'train_size': float/int (required, 0-1 proportion or count)
            'valid_size': float/int (optional, None for train/test only)
            'test_size': float/int (optional, auto-calculated if missing)
            'seed': int (optional, for reproducibility)

    Returns:
        DatasetDict with splits:
            3-way: {'train': Dataset, 'valid': Dataset, 'test': Dataset}
            2-way: {'train': Dataset, 'test': Dataset}

    Raises:
        ValueError: Invalid split sizes or proportions summing > 1.0.
        KeyError: Missing required 'train_size' in config.
    """
    # Extract configuration values
    train_size = split_config.get('train_size')
    valid_size = split_config.get('valid_size')
    test_size = split_config.get('test_size')
    seed = split_config.get('seed')

    # Validate required parameters
    if train_size is None:
        logger.error("'train_size' is required in split_config.")
        raise ValueError("'train_size' must be provided in split_config.")

    # Split: train/valid/test
    if valid_size is not None:
        # If test_size is not provided, calculate it
        if test_size is None:
            if isinstance(train_size, float) and isinstance(valid_size, float):
                # Proportions case
                test_size = 1.0 - train_size - valid_size
                if test_size < 0:
                    logger.error(
                        'train_size (%f) + valid_size (%f) > 1.0',
                        train_size, valid_size
                    )
                    raise ValueError(
                        'train_size + valid_size '
                        'cannot exceed 1.0 for proportions.'
                    )
            else:
                # Absolute numbers case
                total_samples = len(dataset)
                test_size = total_samples - train_size - valid_size
                if test_size < 0:
                    logger.error(
                        'train_size (%f) + valid_size (%f)'
                        ' > total_samples (%d)',
                        train_size, valid_size, total_samples
                    )
                    raise ValueError(
                        'train_size + valid_size cannot exceed total samples.'
                    )

        logger.info(
            'Splitting into 3 sets: '
            'train_size= %f, valid_size=%f, test_size=%f',
            train_size, valid_size, test_size
        )

        # First split: separate train from (valid + test)
        train_test_split = dataset.train_test_split(
            train_size=train_size,
            test_size=(
                1.0 - train_size if isinstance(train_size, float) else None
            ),
            seed=seed,
        )
        # Calculate proportion for the second split
        if isinstance(valid_size, float):
            valid_test_ratio = valid_size / (valid_size + test_size)
        else:
            valid_test_ratio = valid_size / (valid_size + test_size)

        valid_test_split = train_test_split['test'].train_test_split(
            train_size=valid_test_ratio,
            seed=seed,
        )

        new_dataset = DatasetDict({
            'train': train_test_split['train'],
            'valid': valid_test_split['train'],
            'test': valid_test_split['test'],
        })

        logger.info(
            'Dataset split successfully: train=%f, valid=%f, test=%f',
            len(new_dataset['train']), len(new_dataset['valid']),
            len(new_dataset['test'])
        )

    # Split: train/test
    else:
        if test_size is None:
            test_size = (
                1.0 - train_size if isinstance(train_size, float) else None
            )

        logger.info(
            'Splitting into 2 sets: train_size=%f, test_size=%f',
            train_size, test_size
        )

        new_dataset = dataset.train_test_split(
            train_size=train_size,
            test_size=test_size,
            seed=seed,
        )

        logger.info(
            'Dataset split successfully: train=%f, test=%f',
            len(new_dataset['train']), len(new_dataset['test'])
        )

    return new_dataset


def save_to_parquet(
    dataset: Optional[DatasetDict | Dataset],
    save_path: str
) -> None:
    """Save Dataset or DatasetDict to parquet files.

    Handles both single Dataset and DatasetDict (train/valid/test splits).
    Creates 'clean_data' directory automatically.

    Args:
        dataset: Dataset or DatasetDict to save.
        save_path: Base path for saving parquet files.
    """
    os.makedirs('clean_data', exist_ok=True)

    if isinstance(dataset, DatasetDict):
        splits = (
            ['train', 'valid', 'test']
            if 'valid' in dataset
            else ['train', 'test']
        )
        for split in splits:
            dataset[split].to_parquet(
                path_or_buf=f'{save_path}/{split}.parquet'
            )
            logger.info('%s part of dataset was saved to %s', split, save_path)
    else:
        dataset.to_parquet(path_or_buf=f'{save_path}/dataset.parquet')
        logger.info('Dataset was saved to: %s', save_path)


def to_dataset(
    data=None,
    root_path=None,
    split_config=None,
    save_path=None,
    sampling_rate=16000,
) -> Union[Dataset, DatasetDict]:
    """Convert audio data to HuggingFace Dataset with audio features.

    Loads from list of dicts or directory path. Supports optional splitting and
    parquet saving.

    Args:
        data: List of dicts with 'audio' (path) and 'transcription' keys.
        root_path: Directory path to load audio data.
        split_config: Dict for splitting 
          {'train_size': ..., 'valid_size': ..., ...}.
        save_path: Path to save parquet files; None to skip.
        sampling_rate: Target audio sampling rate in Hz.

    Returns:
        Dataset (no split) or DatasetDict (with splits: train/valid/test).

    Raises:
        ValueError: Neither data nor root_path provided.
    """
    # Validate input parameters
    if data is None and root_path is None:
        logger.error('Both data and root_path are None.')
        raise ValueError("Either 'data' or 'root_path' must be provided.")

    # Load data from root_path if provided
    if root_path:
        try:
            data = load_files(root_path)
            logger.info('Loaded %d items from %s', len(data), root_path)
        except Exception as e:
            logger.error('Failed to load data from %s: %s', root_path, e)
            raise

    # Define dataset features
    features = Features({
        'audio': Audio(sampling_rate=sampling_rate),
        'transcription': Value('string')
    })

    # Create HuggingFace Dataset
    try:
        dataset = Dataset.from_list(data, features=features)
        logger.info('Created Dataset with %d entries.', len(dataset))
    except Exception as e:
        logger.error('Failed to create Dataset from data: %s', e)
        raise

    # Split dataset if configuration provided
    if split_config is not None:
        dataset = split_data(dataset, split_config)
        logger.info('Dataset split according to configuration')

    # Save dataset if path provided
    if save_path is not None:
        save_to_parquet(dataset, save_path)
        logger.info('Dataset saved to %s', save_path)
    return dataset


def load_from_parquet(path: str) -> DatasetDict | Dataset:
    """Load parquet dataset splits from directory.

    Searches for train.parquet, valid.parquet, test.parquet. 
    Returns DatasetDict for multiple splits or single Dataset otherwise.

    Args:
        path: Directory with parquet files (train/valid/test.parquet).

    Returns:
        DatasetDict (multi-split) or Dataset (single file).
    """
    splits = ['train', 'valid', 'test']
    data_files = {split: glob.glob(os.path.join(path, f'{split}.parquet'))
                  for split in splits}
    data_files = {k: v for k, v in data_files.items() if v}

    if not data_files:
        raise ValueError('No parquet files found')

    dataset = load_dataset('parquet', data_files=data_files)

    if len(data_files) > 1:
        return dataset
    else:
        return next(iter(dataset.values()))
