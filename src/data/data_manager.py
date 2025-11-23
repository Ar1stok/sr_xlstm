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
    """
    Loads audio file paths and their corresponding transcriptions 
    from a nested directory structure.

    Parameters
    ----------
    root : str
        Root directory path containing subdirectories 
        with audio (.wav) and transcription (.txt) files.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries, each with keys 'audio' (file path) 
        and 'transcription' (text).
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
    """
    Split a HuggingFace Dataset into train, test, and/or validation sets.

    Divides a single Dataset into multiple splits 
    based on provided configuration.
    Supports stratified splitting, custom seed for reproducibility, 
    and flexible split ratios (2-way or 3-way split).

    Parameters
    ----------
    dataset : Dataset
        HuggingFace Dataset to be split.
    split_config : dict
        Configuration dictionary for dataset splitting with the following keys:

        - ``'train_size'`` : float or int
            Proportion (0-1) or absolute number of samples for the training set.
            Required.
        - ``'valid_size'`` : float or int, optional
            Proportion (0-1) or absolute number of samples for the valid set.
            If None, only train/test split is performed. Default is None.
        - ``'test_size'`` : float or int, optional
            Proportion (0-1) or absolute number of samples for the test set.
            If not provided, automatically calculated as remaining samples.
            Default is None.
        - ``'seed'`` : int, optional
            Random seed for reproducibility.

    Returns
    -------
    DatasetDict
        Dictionary containing split datasets. Keys depend on configuration:

        - If ``'valid_size'`` is provided: 
        ``{'train': Dataset, 'valid': Dataset, 'test': Dataset}``
        
        - If ``'valid_size'`` is None: 
        ``{'train': Dataset, 'test': Dataset}``

    Raises
    ------
    ValueError
        If split configuration is invalid (e.g., sizes exceed total samples,
        train_size is not provided, or sizes don't sum to 1.0 for proportions).
    KeyError
        If required keys are missing from `split_config`.

    Examples
    --------
    Split into train/valid/test with equal proportions:

    >>> from datasets import Dataset
    >>> dataset = Dataset.from_dict({'text': ['a', 'b', 'c', 'd', 'e']})
    >>> config = {
    >>>           'train_size': 0.6, 
    >>>           'valid_size': 0.2, 
    >>>           'test_size': 0.2, 
    >>>           'seed': 42
    >>>           }
    >>> splits = split_data(dataset, config)
    >>> len(splits['train']), len(splits['valid']), len(splits['test'])
    (3, 1, 1)

    Split into train/test only:

    >>> config = {'train_size': 0.8, 'test_size': 0.2, 'seed': 42}
    >>> splits = split_data(dataset, config)
    >>> len(splits['train']), len(splits['test'])
    (4, 1)
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
                        'train_size + valid_size ' \
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
            'Splitting into 3 sets: ' \
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
    """Save Dataset in .parquet format"""
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
            logger.info('%d part of dataset was saved to clean_data', split)
    else:
        dataset.to_parquet(path_or_buf=f'{save_path}/dataset.parquet')
        logger.info('Dataset was saved to clean_data')


def to_dataset(
        data=None,
        root_path=None,
        split_config=None,
        save_path=None,
        sampling_rate=16000,
        ) -> Union[Dataset, DatasetDict]:
    """
    Convert audio data into a HuggingFace Dataset with audio features.

    Loads audio data from either a list of dictionaries or a directory path,
    then converts it into a HuggingFace Dataset with specified audio features
    and transcriptions. Optionally splits the dataset and saves to disk.

    Parameters
    ----------
    data : list of dict, optional
        List where each element is a dictionary with keys ``'audio'`` and 
        ``'transcription'``. Either `data` or `root_path` must be provided.
        Default is None.
    root_path : str, optional
        Path to the directory or file containing the audio data. 
        If provided, data is loaded from this path. Either `data` or 
        `root_path` must be provided. Default is None.
    split_config : dict, optional
        Configuration for dataset splitting. If None, no splitting is performed.
        Expected keys:
        
        - ``'train_size'`` : float or int
            Proportion (0-1) or absolute number of samples for training set.
        - ``'valid_size'`` : float or int, optional
            Proportion (0-1) or absolute number of samples for validation set.
        - ``'test_size'`` : float or int, optional
            Proportion (0-1) or absolute number of samples for test set.
        - ``'stratify'`` : str, optional
            Column name to use for stratified splitting.
        - ``'seed'`` : int, optional
            Random seed for reproducibility.
            
        Example: ``{
                    'train_size': 0.8, 
                    'valid_size': 0.1, 
                    'test_size': 0.1, 
                    'seed': 42
                    }``
        Default is None.
    save_path : str, optional
        Path where the dataset should be saved in .parquet format. 
        If None, dataset is not saved to disk. Default is None.
    sampling_rate : int, optional
        Target sampling rate for audio in Hz. Default is 16000.

    Returns
    -------
    Dataset or DatasetDict
        If `split_config` is None, returns a single HuggingFace Dataset.
        If `split_config` is provided, returns a DatasetDict with splits
        (e.g., train, valid, test) as specified in the configuration.

    Examples
    --------
    Create a dataset from a list of dictionaries:
    
    >>> data = [
    ...     {'audio': '/path/to/audio1.wav', 
    ...      'transcription': 'Example of transcription one'},
    ...     {'audio': '/path/to/audio2.wav', 
    ...      'transcription': 'Example of transcription two'}
    ... ]
    >>> dataset = to_dataset(data=data)
    {'audio': Audio(sampling_rate=16000, decode=True),
    'transcription': Value('string')}
    
    Load data from directory and split into train/valid/test:
    
    >>> config = {'train_size': 0.7, 'valid_size': 0.15, 'test_size': 0.15}
    >>> dataset_dict = to_dataset(
    ...     root_path='/path/to/audio_dir',
    ...     split_config=config,
    ...     save_path='/path/to/save'
    ... )
    >>> dataset_dict.keys()
    dict_keys(['train', 'valid', 'test'])
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
    """
    Load parquet dataset splits from given directory.

    Checks for 'train', 'valid', and 'test' parquet files inside `path`
    and loads available from each. Returns a DatasetDict 
    if multiple splits exist, else a single Dataset.

    Parameters
    ----------
    path : str
        Root directory containing 'train', and/or 'valid', 
        and/or 'test' parquet files.

    Returns
    -------
    DatasetDict or Dataset
        Loaded dataset object. DatasetDict if multiple splits are present;
        Otherwise a single Dataset.
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
