from __future__ import annotations
from pathlib import Path
import os
import random
import sys
import json
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LNCRFM_ROOT = os.path.dirname(CURRENT_DIR)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
if LNCRFM_ROOT not in sys.path:
    sys.path.insert(0, LNCRFM_ROOT)
from LncDowntask.data_paths import checkpoint_path, data_path

def _import_lncrfm_modules():
    from src.lncrfm.data.sequence import process_mlm_input_seq
    from src.lncrfm.data.tokenizer import get_mlm_tokenizer
    from src.lncrfm.model import get_lncrfm
    return (process_mlm_input_seq, get_mlm_tokenizer, get_lncrfm)
DEFAULT_HUMAN_MRNA_DMS_DIR = data_path('mRNA-DMS.parquet')
DEFAULT_HUMAN_MRNA_STRUCTURE_DIR = data_path('mRNA-Structure.parquet')
DEFAULT_CHECKPOINT_DIR = checkpoint_path()
LNCRFM_MOTIF_BACKEND = 'lncrfm'

def sanitize_sequence(sequence: str) -> str:
    return ''.join(str(sequence).split()).upper().replace('T', 'U')

def split_sequence_into_windows(sequence: str, window_nt_length: int, overlap: int) -> List[str]:
    sequence = sanitize_sequence(sequence)
    if window_nt_length <= 0:
        raise ValueError(f'window_nt_length must be positive, got {window_nt_length}')
    if len(sequence) <= window_nt_length:
        return [sequence]
    normalized_overlap = max(0, min(int(overlap), window_nt_length - 1))
    stride = max(1, window_nt_length - normalized_overlap)
    windows: List[str] = []
    start = 0
    while start < len(sequence):
        end = min(start + window_nt_length, len(sequence))
        windows.append(sequence[start:end])
        if end >= len(sequence):
            break
        start += stride
    return windows

def build_sequence_token_mask(attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor | None=None) -> torch.Tensor:
    token_mask = attention_mask.bool()
    if special_tokens_mask is not None:
        token_mask = token_mask & ~special_tokens_mask.bool()
    return token_mask

def _model_nt_context_length(tokenizer, max_length: int, model_backend: str) -> int:
    return max(1, int(max_length) - 2)

def prepare_sequence_batch_item(sequence: str, tokenizer, max_length: int, model_backend: str, *, long_seq_strategy: str='truncation', window_overlap: int=256) -> Dict[str, object]:
    sequence = sanitize_sequence(sequence)
    nt_max_length = _model_nt_context_length(tokenizer, max_length, model_backend)
    if long_seq_strategy == 'sliding_window':
        windows = split_sequence_into_windows(sequence, nt_max_length, window_overlap)
        encoded_windows = [encode_sequence(window, tokenizer, max_length, model_backend) for window in windows]
        return {'input_ids': [item['input_ids'] for item in encoded_windows], 'attention_mask': [item['attention_mask'] for item in encoded_windows], 'special_tokens_mask': [item.get('special_tokens_mask') for item in encoded_windows], 'nt_len': [len(window) for window in windows]}
    truncated = sequence[:nt_max_length]
    encoded = encode_sequence(truncated, tokenizer, max_length, model_backend)
    return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'], 'special_tokens_mask': encoded.get('special_tokens_mask'), 'nt_len': len(truncated)}

def collate_sequence_batch(tokenizer, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    first_input_ids = batch[0]['input_ids']
    if isinstance(first_input_ids, list) and first_input_ids and isinstance(first_input_ids[0], list):
        batch_size = len(batch)
        max_windows = max((len(item['input_ids']) for item in batch))
        flat_features: List[Dict[str, List[int]]] = []
        flat_nt_lens: List[int] = []
        valid_windows: List[bool] = []
        pad_token_id = int(getattr(tokenizer, 'pad_token_id', 0) or 0)
        for item in batch:
            num_windows = len(item['input_ids'])
            for input_ids, attention_mask, special_tokens_mask, nt_len in zip(item['input_ids'], item['attention_mask'], item['special_tokens_mask'], item['nt_len']):
                flat_features.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'special_tokens_mask': special_tokens_mask})
                flat_nt_lens.append(int(nt_len))
                valid_windows.append(True)
            for _ in range(max_windows - num_windows):
                flat_features.append({'input_ids': [pad_token_id], 'attention_mask': [0], 'special_tokens_mask': [0]})
                flat_nt_lens.append(0)
                valid_windows.append(False)
        padded = tokenizer.pad(flat_features, return_tensors='pt')
        _, padded_len = padded['input_ids'].shape
        input_ids = padded['input_ids'].view(batch_size, max_windows, padded_len)
        attention_mask = padded['attention_mask'].view(batch_size, max_windows, padded_len)
        special_tokens_mask = padded.get('special_tokens_mask')
        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(padded['attention_mask'])
        token_mask = build_sequence_token_mask(padded['attention_mask'], special_tokens_mask).view(batch_size, max_windows, padded_len)
        window_mask = torch.tensor(valid_windows, dtype=torch.bool).view(batch_size, max_windows)
        nt_lens_tensor = torch.tensor(flat_nt_lens, dtype=torch.long).view(batch_size, max_windows)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_mask': token_mask, 'window_mask': window_mask, 'nt_lens': nt_lens_tensor}
    features = []
    for item in batch:
        feature = {'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']}
        if item.get('special_tokens_mask') is not None:
            feature['special_tokens_mask'] = item['special_tokens_mask']
        features.append(feature)
    padded = tokenizer.pad(features, return_tensors='pt')
    special_tokens_mask = padded.get('special_tokens_mask')
    if special_tokens_mask is None:
        special_tokens_mask = torch.zeros_like(padded['attention_mask'])
    padded['token_mask'] = build_sequence_token_mask(padded['attention_mask'], special_tokens_mask)
    return padded

def pooled_sequence_features(encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_mask: torch.Tensor, *, pooling: str) -> torch.Tensor:
    if pooling not in {'cls', 'mean'}:
        raise ValueError(f'Unsupported pooling mode: {pooling}')
    if input_ids.dim() == 2:
        embeddings = encoder(input_ids=input_ids, attention_mask=attention_mask)
        if pooling == 'cls':
            return embeddings[:, 0, :]
        mask = token_mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if input_ids.dim() != 3:
        raise ValueError(f'Expected input_ids rank 2 or 3, got shape {tuple(input_ids.shape)}')
    batch_size, num_windows, seq_len = input_ids.shape
    flat_input_ids = input_ids.reshape(batch_size * num_windows, seq_len)
    flat_attention_mask = attention_mask.reshape(batch_size * num_windows, seq_len)
    flat_token_mask = token_mask.reshape(batch_size * num_windows, seq_len)
    embeddings = encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
    if pooling == 'cls':
        pooled = embeddings[:, 0, :]
    else:
        mask = flat_token_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    pooled = pooled.view(batch_size, num_windows, -1)
    valid_windows = token_mask.any(dim=-1).unsqueeze(-1).to(pooled.dtype)
    return (pooled * valid_windows).sum(dim=1) / valid_windows.sum(dim=1).clamp_min(1.0)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pearson_corr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.float()
    targets = targets.float()
    if preds.numel() <= 1:
        return 0.0
    preds = preds - preds.mean()
    targets = targets - targets.mean()
    denom = preds.norm() * targets.norm()
    if denom.item() == 0:
        return 0.0
    return float((preds * targets).sum().item() / denom.item())

def can_form_canonical_pair(left_base: str, right_base: str, allow_wobble: bool=True) -> bool:
    pair = (left_base, right_base)
    if pair in {('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C')}:
        return True
    return allow_wobble and pair in {('G', 'U'), ('U', 'G')}

def build_tokenizer(args):
    _, get_mlm_tokenizer, _ = _import_lncrfm_modules()
    checkpoint = Path(args.from_pretrained or checkpoint_path())
    options = json.loads((checkpoint / 'model_config.json').read_text())['factory_kwargs']
    max_length = min(args.max_length, options['max_length'])
    tokenizer = get_mlm_tokenizer(tokenizer_file=str(checkpoint / 'tokenizer.json'), max_length=max_length)
    return (tokenizer, max_length)

def build_lncrfm_encoder(args, tokenizer):
    _, _, get_lncrfm = _import_lncrfm_modules()
    checkpoint = Path(args.from_pretrained or checkpoint_path())
    options = json.loads((checkpoint / 'model_config.json').read_text())['factory_kwargs']
    args.dim = options['dim']
    args.layer = options['layer']
    model = get_lncrfm(**options, from_pretrained=str(checkpoint), tokenizer=tokenizer, dtype=torch.float32)
    if args.freeze_base or getattr(args, 'freeze_encoder', False):
        for parameter in model.parameters():
            parameter.requires_grad = False
    return (model, args.dim)

class LncRFMEncoderWrapper(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states[-1]

def configure_runtime_precision(args) -> None:
    requested = getattr(args, 'allow_tf32', None)
    if requested is None:
        requested = os.getenv('LNCRFM_ALLOW_TF32')
    if requested is None:
        return
    enabled = requested if isinstance(requested, bool) else str(requested).lower() in {'1', 'true', 'yes'}
    torch.backends.cuda.matmul.allow_tf32 = enabled
    if enabled:
        torch.backends.cudnn.allow_tf32 = True
    args.allow_tf32 = enabled
    args.window_batch_size = int(getattr(args, 'window_batch_size', os.getenv('LNCRFM_WINDOW_BATCH_SIZE', '0')))
    args.window_activation_checkpoint = bool(getattr(args, 'window_activation_checkpoint', os.getenv('LNCRFM_WINDOW_CHECKPOINT', '0') == '1'))
    args.window_checkpoint_encoder = bool(getattr(args, 'window_checkpoint_encoder', os.getenv('LNCRFM_WINDOW_CHECKPOINT_ENCODER', '0') == '1'))
    os.environ['LNCRFM_WINDOW_BATCH_SIZE'] = str(args.window_batch_size)
    os.environ['LNCRFM_WINDOW_CHECKPOINT'] = '1' if args.window_activation_checkpoint else '0'
    os.environ['LNCRFM_WINDOW_CHECKPOINT_ENCODER'] = '1' if args.window_checkpoint_encoder else '0'
    print({'float32_matmul_tf32': enabled, 'float32_cudnn_tf32': torch.backends.cudnn.allow_tf32, 'window_batch_size': args.window_batch_size, 'window_activation_checkpoint': args.window_activation_checkpoint, 'window_checkpoint_encoder': args.window_checkpoint_encoder}, flush=True)

def build_encoder(args, tokenizer, device: torch.device) -> Tuple[nn.Module, int]:
    configure_runtime_precision(args)
    encoder_model, hidden_size = build_lncrfm_encoder(args, tokenizer)
    encoder = LncRFMEncoderWrapper(encoder_model)
    return (encoder.to(device), hidden_size)

def encode_sequence(sequence: str, tokenizer, max_length: int, model_backend: str) -> Dict[str, List[int]]:
    sequence = sanitize_sequence(sequence)
    process_mlm_input_seq, _, _ = _import_lncrfm_modules()
    text = process_mlm_input_seq(sequence, tokenizer.bos_token, tokenizer.eos_token)
    return tokenizer(text, truncation=True, max_length=max_length, padding=False, return_attention_mask=True, return_special_tokens_mask=True)

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

def precision_recall_f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    if precision + recall == 0:
        return (precision, recall, 0.0)
    return (precision, recall, 2.0 * precision * recall / (precision + recall))

def build_default_thresholds() -> List[float]:
    return [round(step / 20.0, 2) for step in range(1, 20)]
