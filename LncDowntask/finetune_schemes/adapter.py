from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
AdapterStrategy = Literal['motif_structure']
__all__ = ['AdapterConfig', 'MotifStructureAdapter', 'build_adapter', 'masked_mean_pool']

def _as_tuple(values: Sequence[int] | int) -> Tuple[int, ...]:
    if isinstance(values, int):
        return (values,)
    return tuple((int(v) for v in values))

def _validate_kernel_sizes(kernel_sizes: Sequence[int]) -> Tuple[int, ...]:
    ks = _as_tuple(kernel_sizes)
    if not ks:
        raise ValueError('kernel_sizes must not be empty')
    for kernel_size in ks:
        if kernel_size <= 0:
            raise ValueError(f'kernel size must be positive, got {kernel_size}')
        if kernel_size % 2 == 0:
            raise ValueError(f'kernel size must be odd to preserve sequence length, got {kernel_size}')
    return ks

def _ensure_bool_mask(mask: Optional[Tensor], hidden_states: Tensor) -> Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    if mask is None:
        return torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=torch.bool)
    if mask.dim() != 2:
        raise ValueError(f'attention_mask must have shape [batch, length], got {tuple(mask.shape)}')
    if mask.shape[:2] != (batch_size, seq_len):
        raise ValueError(f'attention_mask shape must match hidden_states shape [{batch_size}, {seq_len}], got {tuple(mask.shape)}')
    return mask.to(dtype=torch.bool)

def masked_mean_pool(hidden_states: Tensor, attention_mask: Optional[Tensor]=None) -> Tensor:
    mask = _ensure_bool_mask(attention_mask, hidden_states)
    weights = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    summed = (hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom

@dataclass
class AdapterConfig:
    hidden_size: int
    bottleneck_size: Optional[int] = None
    dropout: float = 0.1
    residual_scale: float = 1.0
    use_layernorm: bool = True
    strategy: AdapterStrategy = 'motif_structure'
    local_kernel_sizes: Tuple[int, ...] = (3, 5)
    structure_dilations: Tuple[int, ...] = (2, 4)
    router_hidden_size: Optional[int] = None
    router_temperature: float = 1.0

    def resolved_bottleneck_size(self) -> int:
        if self.bottleneck_size is not None:
            return int(self.bottleneck_size)
        return max(32, min(128, self.hidden_size // 2))

    def resolved_router_hidden_size(self) -> int:
        if self.router_hidden_size is not None:
            return int(self.router_hidden_size)
        return max(32, min(128, self.resolved_bottleneck_size()))

class _MaskedDepthwiseConv(nn.Module):

    def __init__(self, channels: int, kernel_size: int, dilation: int=1, dropout: float=0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd to preserve sequence length')
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=channels, bias=True)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        mask = _ensure_bool_mask(attention_mask, hidden_states)
        x = hidden_states.transpose(1, 2)
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.dropout(x).transpose(1, 2)
        x = self.norm(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

class _GlobalContextBranch(nn.Module):

    def __init__(self, hidden_size: int, bottleneck_size: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(hidden_size, bottleneck_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(bottleneck_size, bottleneck_size))
        self.norm = nn.LayerNorm(bottleneck_size)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        context = masked_mean_pool(hidden_states, attention_mask)
        context = self.proj(context)
        context = self.norm(context)
        return context.unsqueeze(1).expand(-1, hidden_states.size(1), -1)

class _BaseAdapter(nn.Module):
    num_branches: int = 3

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.bottleneck_size = config.resolved_bottleneck_size()
        self.dropout = nn.Dropout(config.dropout)
        self.residual_scale = float(config.residual_scale)
        self.temperature = max(float(config.router_temperature), 1e-05)
        self.pre_norm = nn.LayerNorm(self.hidden_size) if config.use_layernorm else nn.Identity()
        self.output_norm = nn.LayerNorm(self.hidden_size) if config.use_layernorm else nn.Identity()
        self.down_proj = nn.Linear(self.hidden_size, self.bottleneck_size)
        self.up_proj = nn.Linear(self.bottleneck_size, self.hidden_size)
        self.router_hidden_size = config.resolved_router_hidden_size()

    def _mix_branches(self, branch_outputs: Sequence[Tensor], router_features: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, token_embeddings: Tensor, attention_mask: Optional[Tensor]=None) -> Tensor:
        if token_embeddings.dim() != 3:
            raise ValueError(f'token_embeddings must have shape [batch, length, hidden], got {tuple(token_embeddings.shape)}')
        mask = _ensure_bool_mask(attention_mask, token_embeddings)
        residual = token_embeddings
        x = self.pre_norm(token_embeddings)
        x = self.down_proj(x)
        branches = self._branch_outputs(x, mask)
        router_features = self._router_features(x, mask)
        mixed = self._mix_branches(branches, router_features)
        mixed = self.up_proj(self.dropout(mixed))
        mixed = self.output_norm(residual + self.residual_scale * self.dropout(mixed))
        mixed = mixed * mask.unsqueeze(-1).to(dtype=mixed.dtype)
        return mixed

    def _branch_outputs(self, hidden_states: Tensor, attention_mask: Tensor) -> Sequence[Tensor]:
        raise NotImplementedError

class MotifStructureAdapter(_BaseAdapter):

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        kernel_sizes = _validate_kernel_sizes(config.local_kernel_sizes)
        self.motif_branch = nn.ModuleList((_MaskedDepthwiseConv(self.bottleneck_size, kernel_size=kernel, dilation=1, dropout=config.dropout) for kernel in kernel_sizes))
        self.structure_branch = nn.ModuleList((_MaskedDepthwiseConv(self.bottleneck_size, kernel_size=3, dilation=dilation, dropout=config.dropout) for dilation in _as_tuple(config.structure_dilations)))
        self.global_branch = _GlobalContextBranch(self.bottleneck_size, self.bottleneck_size, config.dropout)
        self.token_gate = nn.Sequential(nn.Linear(self.bottleneck_size, self.bottleneck_size), nn.GELU(), nn.Linear(self.bottleneck_size, 1))
        self.router = nn.Sequential(nn.Linear(self.bottleneck_size, self.router_hidden_size), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(self.router_hidden_size, 3))

    def _router_features(self, hidden_states: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        return masked_mean_pool(hidden_states, attention_mask)

    def _branch_outputs(self, hidden_states: Tensor, attention_mask: Tensor) -> Sequence[Tensor]:
        motif = sum((branch(hidden_states, attention_mask) for branch in self.motif_branch)) / len(self.motif_branch)
        structural = sum((branch(hidden_states, attention_mask) for branch in self.structure_branch)) / len(self.structure_branch)
        global_context = self.global_branch(hidden_states, attention_mask)
        return [motif, structural, global_context]

    def _mix_branches(self, branch_outputs: Sequence[Tensor], router_features: Tensor) -> Tensor:
        logits = self.router(router_features)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        motif, structural, global_context = branch_outputs
        token_gate = torch.sigmoid(self.token_gate(motif + structural))
        blended_local = 0.5 * (motif + structural)
        mixed = token_gate * blended_local + (1.0 - token_gate) * global_context
        mixed = mixed * weights[:, 0].view(-1, 1, 1) + structural * weights[:, 1].view(-1, 1, 1) + motif * weights[:, 2].view(-1, 1, 1)
        return mixed

def build_adapter(config: AdapterConfig) -> nn.Module:
    return MotifStructureAdapter(config)
