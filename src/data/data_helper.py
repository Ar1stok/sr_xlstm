from typing import Dict, List
from datasets import Dataset, Features, Value, Audio
from pathlib import Path


def load_files(root: str) -> List[Dict[str, str]]:
    """
    Loads audio file paths and their corresponding transcriptions from a nested directory structure.

    Parameters
    ----------
    root : str
        Root directory path containing subdirectories with audio (.wav) and transcription (.txt) files.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries, each with keys 'audio' (file path) and 'transcription' (text).
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


def to_dataset(data: list) -> Dataset:
    """
    Converts a list of dictionaries with audio paths and transcriptions into a HuggingFace Dataset.

    Parameters
    ----------
    data : List[Dict[str, str]]
        A list where each element is a dictionary with keys 'audio' and 'transcription'.

    Returns
    -------
    Dataset
        HuggingFace Dataset with features 'audio' (Audio feature with 16000 Hz) and 'transcription' (string).
    """

    features = Features({
        'audio': Audio(sampling_rate=16000),
        'transcription': Value('string')
    })

    dataset = Dataset.from_list(data, features=features)
    return dataset