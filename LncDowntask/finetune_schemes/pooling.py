from __future__ import annotations
from typing import Optional
import torch

def _ensure_3d_embeddings(token_embeddings: torch.Tensor) -> torch.Tensor:
    if token_embeddings.dim() != 3:
        raise ValueError(f'token_embeddings must have shape [batch, length, hidden_size], got {tuple(token_embeddings.shape)}')
    if not token_embeddings.is_floating_point():
        raise ValueError('token_embeddings must be a floating-point tensor.')
    return token_embeddings

def _normalize_attention_mask(attention_mask: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = reference.shape[:2]
    if attention_mask is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=reference.device)
    if attention_mask.dim() == 3 and attention_mask.size(-1) == 1:
        attention_mask = attention_mask.squeeze(-1)
    if attention_mask.dim() != 2:
        raise ValueError(f'attention_mask must have shape [batch, length] or [batch, length, 1], got {tuple(attention_mask.shape)}')
    if attention_mask.shape[0] != batch_size or attention_mask.shape[1] != seq_len:
        raise ValueError(f'attention_mask shape must match token_embeddings batch and length dimensions, got {tuple(attention_mask.shape)} vs {tuple(reference.shape)}')
    return attention_mask.to(device=reference.device, dtype=torch.bool)

def masked_mean_pool(token_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    token_embeddings = _ensure_3d_embeddings(token_embeddings)
    mask = _normalize_attention_mask(attention_mask, token_embeddings).unsqueeze(-1)
    weights = mask.to(dtype=token_embeddings.dtype)
    numerator = (token_embeddings * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return numerator / denominator

def masked_cls_pool(token_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, *, cls_index: int=0) -> torch.Tensor:
    token_embeddings = _ensure_3d_embeddings(token_embeddings)
    if cls_index < 0 or cls_index >= token_embeddings.size(1):
        raise ValueError(f'cls_index {cls_index} is out of range for length {token_embeddings.size(1)}')
    return token_embeddings[:, cls_index, :]
