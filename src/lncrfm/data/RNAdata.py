import os
from pathlib import Path
from datasets import load_dataset
from .sequence import process_mlm_input_seq

def preprocess_mlm_with_structure(samples, tokenizer):
    processed = {'input_ids': [], 'attention_mask': [], 'connects': [], 'length': []}
    for sequence, partners in zip(samples['seq'], samples['connects']):
        if len(partners) != len(sequence) + 2:
            raise ValueError('Pairing vectors must include leading and trailing boundary slots')
        sequence = sequence[:tokenizer.model_max_length - 2]
        tokens = tokenizer(process_mlm_input_seq(sequence), padding=False, truncation=False)
        length = len(tokens['input_ids'])
        connects = [int(partner) if 0 < partner < length - 1 else 0 for partner in partners[:length]]
        connects[0] = connects[-1] = 0
        processed['input_ids'].append(tokens['input_ids'])
        processed['attention_mask'].append(tokens['attention_mask'])
        processed['connects'].append(connects)
        processed['length'].append(length)
    return processed

def preprocess_and_load_dataset(data_path, tokenizer, min_seq_len=200, max_seq_len=2048):
    source = Path(data_path)
    if (source / 'data').is_dir():
        source = source / 'data'
    paths = [source] if source.is_file() else sorted(source.glob('*.parquet'))
    if not paths or any((path.suffix != '.parquet' for path in paths)):
        raise ValueError('LncCorpus input must contain Parquet files')
    data_files = {}
    for path in paths:
        split = 'test' if path.name.startswith('LncCorpus_heldout_') else 'train'
        data_files.setdefault(split, []).append(str(path))
    dataset = load_dataset('parquet', data_files=data_files)
    for partition in dataset.values():
        if not {'seq', 'connects'}.issubset(partition.column_names):
            raise ValueError('LncCorpus requires seq and connects columns')
    if min_seq_len < 0 or max_seq_len < min_seq_len:
        raise ValueError('Invalid sequence length range')
    workers = int(os.getenv('LNCRFM_PREPROCESS_WORKERS', '1'))
    if workers < 1:
        raise ValueError('LNCRFM_PREPROCESS_WORKERS must be positive')
    dataset = dataset.filter(lambda row: min_seq_len <= len(row['seq']) <= max_seq_len, num_proc=workers)
    if 'train' not in dataset or len(dataset['train']) == 0:
        raise ValueError('No pretraining records in the requested length range')
    return dataset.map(lambda batch: preprocess_mlm_with_structure(batch, tokenizer), batched=True, num_proc=workers, remove_columns=['seq', 'connects'])
