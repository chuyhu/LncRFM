from __future__ import annotations
import argparse
import os
import time
from typing import Iterable, Tuple
import torch
from torch import Tensor, nn
from torch.nn.parameter import UninitializedParameter
from torch.utils.checkpoint import checkpoint
from finetune_schemes.adapter import AdapterConfig, build_adapter
from finetune_schemes.pooling import masked_cls_pool, masked_mean_pool
SUPPORTED_FINETUNE_SCHEMES = ('baseline', 'adapter')
DEFAULT_SCHEME_KERNEL_SIZES = '3,5,7'
DEFAULT_SCHEME_STRUCTURE_DILATIONS = '2,4'

def count_initialized_parameters(module: nn.Module, *, trainable_only: bool=False) -> int:
    total = 0
    for param in module.parameters():
        if trainable_only and (not param.requires_grad):
            continue
        if isinstance(param, UninitializedParameter):
            continue
        total += param.numel()
    return total

def _parse_csv_items(raw: str | None) -> Tuple[str, ...]:
    if raw is None:
        return tuple()
    return tuple((item.strip() for item in str(raw).replace(';', ',').split(',') if item.strip()))

def _parse_int_csv(raw: str | None) -> Tuple[int, ...]:
    return tuple((int(item) for item in _parse_csv_items(raw)))

def _resolved_scheme_dropout(args: argparse.Namespace) -> float:
    scheme_dropout = float(getattr(args, 'scheme_dropout', -1.0))
    return float(args.dropout) if scheme_dropout < 0 else scheme_dropout

def _as_bool_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    if mask is None:
        return torch.ones(reference.shape[:-1], dtype=torch.bool, device=reference.device)
    return mask.to(device=reference.device, dtype=torch.bool)

def _as_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)

def _encode_tokens(encoder: nn.Module, input_ids: Tensor, attention_mask: Tensor, *, freeze_encoder: bool) -> Tensor:
    if freeze_encoder:
        with torch.no_grad():
            return encoder(input_ids=input_ids, attention_mask=attention_mask)
    return encoder(input_ids=input_ids, attention_mask=attention_mask)

def _flatten_windows(tensor: Tensor) -> Tuple[Tensor, Tuple[int, ...] | None]:
    if tensor.dim() < 3:
        return (tensor, None)
    if tensor.dim() == 3:
        return (tensor, None)
    if tensor.dim() != 4:
        raise ValueError(f'Expected rank-3 or rank-4 tensor, got shape {tuple(tensor.shape)}')
    batch_size, num_windows = tensor.shape[:2]
    return (tensor.reshape(batch_size * num_windows, *tensor.shape[2:]), (batch_size, num_windows))

def _flatten_ids(input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor, Tuple[int, int] | None]:
    if input_ids.dim() == 2:
        return (input_ids, attention_mask, None)
    if input_ids.dim() != 3:
        raise ValueError(f'Expected input_ids rank 2 or 3, got {tuple(input_ids.shape)}')
    batch_size, num_windows, seq_len = input_ids.shape
    return (input_ids.reshape(batch_size * num_windows, seq_len), attention_mask.reshape(batch_size * num_windows, seq_len), (batch_size, num_windows))

def _average_window_features(features: Tensor, token_mask: Tensor, batch_size: int, num_windows: int) -> Tensor:
    features = features.view(batch_size, num_windows, -1)
    valid_windows = token_mask.view(batch_size, num_windows, token_mask.size(-1)).any(dim=-1).to(features.dtype)
    return (features * valid_windows.unsqueeze(-1)).sum(dim=1) / valid_windows.sum(dim=1, keepdim=True).clamp_min(1.0)

def add_finetune_scheme_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--finetune_scheme', choices=['baseline', 'adapter'], default='baseline')
    parser.add_argument('--scheme_dropout', type=float, default=-1.0)
    parser.add_argument('--scheme_bottleneck_size', type=int, default=192)
    parser.add_argument('--router_hidden_size', type=int, default=128)
    parser.add_argument('--routing_temperature', type=float, default=1.0)
    parser.add_argument('--scheme_kernel_sizes', type=str, default=DEFAULT_SCHEME_KERNEL_SIZES)
    parser.add_argument('--scheme_structure_dilations', type=str, default=DEFAULT_SCHEME_STRUCTURE_DILATIONS)
    parser.add_argument('--adapter_residual_scale', type=float, default=1.0)

class SequenceSchemeEncoder(nn.Module):

    def _report_window_progress(self, batch_size: int, windows: int) -> None:
        every = int(os.getenv('LNCRFM_WINDOW_PROGRESS_EVERY', '0'))
        if every <= 0:
            return
        if not self.training or not torch.is_grad_enabled():
            self._window_progress_state = None
            return
        state = getattr(self, '_window_progress_state', None)
        if state is None:
            self._window_progress_epoch = getattr(self, '_window_progress_epoch', 0) + 1
            state = {'started': time.perf_counter(), 'batches': 0, 'samples': 0, 'first_batch_size': batch_size, 'windows': 0}
            self._window_progress_state = state
        state['batches'] += 1
        state['samples'] += batch_size
        state['windows'] += windows
        if state['batches'] == 1 or state['batches'] % every == 0:
            elapsed = time.perf_counter() - state['started']
            rate = (state['samples'] - state['first_batch_size']) / elapsed if state['batches'] > 1 and elapsed > 0 else None
            print({'window_progress': True, 'epoch_in_process': self._window_progress_epoch, 'train_batch': state['batches'], 'samples_seen': state['samples'], 'windows_seen': state['windows'], 'elapsed_train_s': round(elapsed, 2), 'samples_per_second': round(rate, 3) if rate is not None else None}, flush=True)

    def __init__(self, hidden_size: int, *, finetune_scheme: str='baseline', scheme_dropout: float=0.1, scheme_bottleneck_size: int=192, router_hidden_size: int=128, routing_temperature: float=1.0, kernel_sizes: Iterable[int]=(3, 5, 7), structure_dilations: Iterable[int]=(2, 4), adapter_residual_scale: float=1.0, freeze_encoder: bool=False) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.finetune_scheme = str(finetune_scheme)
        self.freeze_encoder = bool(freeze_encoder)
        if self.finetune_scheme == 'adapter':
            self.scheme_adapter = build_adapter(AdapterConfig(hidden_size=self.hidden_size, bottleneck_size=scheme_bottleneck_size, dropout=scheme_dropout, residual_scale=adapter_residual_scale, strategy='motif_structure', local_kernel_sizes=tuple(kernel_sizes), structure_dilations=tuple(structure_dilations), router_hidden_size=router_hidden_size, router_temperature=routing_temperature))

    def _per_sequence_feature(self, token_embeddings: Tensor, attention_mask: Tensor | None, token_mask: Tensor | None, *, pooling: str) -> Tensor:
        attention_bool = _as_bool_mask(attention_mask, token_embeddings)
        token_bool = _as_bool_mask(token_mask, token_embeddings)
        if self.finetune_scheme == 'baseline':
            if pooling == 'mean':
                return masked_mean_pool(token_embeddings, token_bool)
            return masked_cls_pool(token_embeddings)
        if self.finetune_scheme == 'adapter':
            adapted = self.scheme_adapter(token_embeddings, attention_mask=attention_bool)
            if pooling == 'mean':
                return masked_mean_pool(adapted, token_bool)
            return masked_cls_pool(adapted)
        raise ValueError(f'Unsupported finetune_scheme: {self.finetune_scheme}')

    def forward(self, encoder: nn.Module, input_ids: Tensor, attention_mask: Tensor, token_mask: Tensor, *, pooling: str) -> Tensor:
        flat_input_ids, flat_attention_mask, window_meta = _flatten_ids(input_ids, attention_mask)
        flat_token_mask, _ = _flatten_windows(token_mask.unsqueeze(-1))
        flat_token_mask = flat_token_mask.squeeze(-1)
        window_batch_size = int(os.getenv('LNCRFM_WINDOW_BATCH_SIZE', '0'))
        if window_meta is not None and window_batch_size > 0:
            batch_size, num_windows = window_meta
            valid_indices = flat_token_mask.bool().any(dim=-1).nonzero(as_tuple=True)[0]

            def encode_window_features(ids, attention, tokens):
                embeddings = _encode_tokens(encoder, ids, attention, freeze_encoder=self.freeze_encoder)
                return self._per_sequence_feature(embeddings, attention, tokens, pooling=pooling)

            def adapt_window_features(embeddings, attention, tokens):
                return self._per_sequence_feature(embeddings, attention, tokens, pooling=pooling)
            parts = []
            use_checkpoint = self.training and torch.is_grad_enabled() and (os.getenv('LNCRFM_WINDOW_CHECKPOINT', '0') == '1')
            for indices in valid_indices.split(window_batch_size):
                inputs = (flat_input_ids[indices], flat_attention_mask[indices], flat_token_mask[indices])
                if use_checkpoint:
                    checkpoint_encoder = os.getenv('LNCRFM_WINDOW_CHECKPOINT_ENCODER', '0') == '1'
                    if self.freeze_encoder and (not checkpoint_encoder):
                        embeddings = _encode_tokens(encoder, inputs[0], inputs[1], freeze_encoder=True)
                        features = checkpoint(adapt_window_features, embeddings, *inputs[1:], use_reentrant=False, preserve_rng_state=True)
                    else:
                        features = checkpoint(encode_window_features, *inputs, use_reentrant=False, preserve_rng_state=True)
                else:
                    features = encode_window_features(*inputs)
                parts.append(features)
            if not parts:
                raise ValueError('Sliding-window batch contains no valid windows')
            features = torch.cat(parts, dim=0)
            owners = torch.div(valid_indices, num_windows, rounding_mode='floor')
            sums = features.new_zeros((batch_size, features.size(-1))).index_add(0, owners, features)
            counts = torch.bincount(owners, minlength=batch_size).clamp_min(1).to(features.dtype)
            self._report_window_progress(batch_size, valid_indices.numel())
            return sums / counts.unsqueeze(-1)
        token_embeddings = _encode_tokens(encoder, flat_input_ids, flat_attention_mask, freeze_encoder=self.freeze_encoder)
        features = self._per_sequence_feature(token_embeddings, flat_attention_mask, flat_token_mask, pooling=pooling)
        if window_meta is None:
            return features
        batch_size, num_windows = window_meta
        return _average_window_features(features, flat_token_mask, batch_size, num_windows)

class TokenSchemeEncoder(nn.Module):

    def __init__(self, hidden_size: int, *, finetune_scheme: str='baseline', scheme_dropout: float=0.1, scheme_bottleneck_size: int=192, router_hidden_size: int=128, routing_temperature: float=1.0, kernel_sizes: Iterable[int]=(3, 5, 7), structure_dilations: Iterable[int]=(2, 4), adapter_residual_scale: float=1.0, freeze_encoder: bool=False) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.finetune_scheme = str(finetune_scheme)
        self.freeze_encoder = bool(freeze_encoder)
        if self.finetune_scheme == 'adapter':
            self.scheme_adapter = build_adapter(AdapterConfig(hidden_size=self.hidden_size, bottleneck_size=scheme_bottleneck_size, dropout=scheme_dropout, residual_scale=adapter_residual_scale, strategy='motif_structure', local_kernel_sizes=tuple(kernel_sizes), structure_dilations=tuple(structure_dilations), router_hidden_size=router_hidden_size, router_temperature=routing_temperature))

    def _transform(self, token_embeddings: Tensor, attention_mask: Tensor | None) -> Tensor:
        attention_bool = _as_bool_mask(attention_mask, token_embeddings)
        if self.finetune_scheme == 'baseline':
            return token_embeddings
        if self.finetune_scheme == 'adapter':
            return self.scheme_adapter(token_embeddings, attention_mask=attention_bool)
        raise ValueError(f'Unsupported finetune_scheme: {self.finetune_scheme}')

    def forward(self, encoder: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        flat_input_ids, flat_attention_mask, window_meta = _flatten_ids(input_ids, attention_mask)
        token_embeddings = _encode_tokens(encoder, flat_input_ids, flat_attention_mask, freeze_encoder=self.freeze_encoder)
        transformed = self._transform(token_embeddings, flat_attention_mask)
        if window_meta is None:
            return transformed
        batch_size, num_windows = window_meta
        return transformed.view(batch_size, num_windows, *transformed.shape[1:])

class StructureSchemeEncoder(nn.Module):

    def __init__(self, hidden_size: int, pair_hidden_size: int, *, finetune_scheme: str='baseline', scheme_dropout: float=0.1, scheme_bottleneck_size: int=192, router_hidden_size: int=128, routing_temperature: float=1.0, kernel_sizes: Iterable[int]=(3, 5, 7), structure_dilations: Iterable[int]=(2, 4), adapter_residual_scale: float=1.0, freeze_encoder: bool=False) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.pair_hidden_size = int(pair_hidden_size)
        self.finetune_scheme = str(finetune_scheme)
        self.freeze_encoder = bool(freeze_encoder)
        if self.finetune_scheme == 'adapter':
            self.scheme_adapter = build_adapter(AdapterConfig(hidden_size=self.hidden_size, bottleneck_size=scheme_bottleneck_size, dropout=scheme_dropout, residual_scale=adapter_residual_scale, strategy='motif_structure', local_kernel_sizes=tuple(kernel_sizes), structure_dilations=tuple(structure_dilations), router_hidden_size=router_hidden_size, router_temperature=routing_temperature))

    def forward(self, encoder: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor | None, Tensor | None]:
        flat_input_ids, flat_attention_mask, window_meta = _flatten_ids(input_ids, attention_mask)
        token_embeddings = _encode_tokens(encoder, flat_input_ids, flat_attention_mask, freeze_encoder=self.freeze_encoder)
        attention_bool = _as_bool_mask(flat_attention_mask, token_embeddings)
        pairwise_features = None
        if self.finetune_scheme == 'baseline':
            transformed = token_embeddings
        elif self.finetune_scheme == 'adapter':
            transformed = self.scheme_adapter(token_embeddings, attention_mask=attention_bool)
        else:
            raise ValueError(f'Unsupported finetune_scheme: {self.finetune_scheme}')
        if window_meta is None:
            return (transformed, pairwise_features)
        batch_size, num_windows = window_meta
        if transformed is not None:
            transformed = transformed.view(batch_size, num_windows, *transformed.shape[1:])
        if pairwise_features is not None:
            pairwise_features = pairwise_features.view(batch_size, num_windows, pairwise_features.size(1), pairwise_features.size(2), pairwise_features.size(3))
        return (transformed, pairwise_features)

def build_sequence_scheme_encoder(hidden_size: int, args: argparse.Namespace) -> SequenceSchemeEncoder:
    return SequenceSchemeEncoder(hidden_size, finetune_scheme=getattr(args, 'finetune_scheme', 'baseline'), scheme_dropout=_resolved_scheme_dropout(args), scheme_bottleneck_size=getattr(args, 'scheme_bottleneck_size', 192), router_hidden_size=getattr(args, 'router_hidden_size', 128), routing_temperature=getattr(args, 'routing_temperature', 1.0), kernel_sizes=_parse_int_csv(getattr(args, 'scheme_kernel_sizes', DEFAULT_SCHEME_KERNEL_SIZES)), structure_dilations=_parse_int_csv(getattr(args, 'scheme_structure_dilations', DEFAULT_SCHEME_STRUCTURE_DILATIONS)), adapter_residual_scale=getattr(args, 'adapter_residual_scale', 1.0), freeze_encoder=_as_bool_flag(getattr(args, 'freeze_encoder', False)))

def build_token_scheme_encoder(hidden_size: int, args: argparse.Namespace) -> TokenSchemeEncoder:
    return TokenSchemeEncoder(hidden_size, finetune_scheme=getattr(args, 'finetune_scheme', 'baseline'), scheme_dropout=_resolved_scheme_dropout(args), scheme_bottleneck_size=getattr(args, 'scheme_bottleneck_size', 192), router_hidden_size=getattr(args, 'router_hidden_size', 128), routing_temperature=getattr(args, 'routing_temperature', 1.0), kernel_sizes=_parse_int_csv(getattr(args, 'scheme_kernel_sizes', DEFAULT_SCHEME_KERNEL_SIZES)), structure_dilations=_parse_int_csv(getattr(args, 'scheme_structure_dilations', DEFAULT_SCHEME_STRUCTURE_DILATIONS)), adapter_residual_scale=getattr(args, 'adapter_residual_scale', 1.0), freeze_encoder=_as_bool_flag(getattr(args, 'freeze_encoder', False)))

def build_structure_scheme_encoder(hidden_size: int, pair_hidden_size: int, args: argparse.Namespace) -> StructureSchemeEncoder:
    return StructureSchemeEncoder(hidden_size, pair_hidden_size, finetune_scheme=getattr(args, 'finetune_scheme', 'baseline'), scheme_dropout=_resolved_scheme_dropout(args), scheme_bottleneck_size=getattr(args, 'scheme_bottleneck_size', 192), router_hidden_size=getattr(args, 'router_hidden_size', 128), routing_temperature=getattr(args, 'routing_temperature', 1.0), kernel_sizes=_parse_int_csv(getattr(args, 'scheme_kernel_sizes', DEFAULT_SCHEME_KERNEL_SIZES)), structure_dilations=_parse_int_csv(getattr(args, 'scheme_structure_dilations', DEFAULT_SCHEME_STRUCTURE_DILATIONS)), adapter_residual_scale=getattr(args, 'adapter_residual_scale', 1.0), freeze_encoder=_as_bool_flag(getattr(args, 'freeze_encoder', False)))
