import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def data_path(*parts):
    root = Path(os.environ.get('LNCRFM_DATA_ROOT', PROJECT_ROOT / 'LncBench'))
    if (root / 'data').is_dir():
        root = root / 'data'
    return str(root.joinpath(*parts))

def checkpoint_path():
    return os.environ.get('LNCRFM_CHECKPOINT', str(PROJECT_ROOT / 'checkpoints/lncrfm_pretrained'))
