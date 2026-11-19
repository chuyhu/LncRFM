# Copyright (c) 2023, Albert Gu, Tri Dao.
# Adapted and modified for LncRFM; see LICENSE.

from typing import Optional
from dataclasses import dataclass, field
from functools import partial
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import PretrainedConfig
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan

def _apply_attention_mask(hidden_states, attention_mask):
    if attention_mask is None:
        return hidden_states
    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    return hidden_states * mask

def _reverse_right_padded_sequence(hidden_states, attention_mask):
    if attention_mask is None:
        return torch.flip(hidden_states, dims=[1])
    mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    batch_size, seq_len, dim = hidden_states.shape
    lengths = mask.long().sum(dim=1)
    positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, seq_len)
    reversed_positions = lengths.unsqueeze(1) - 1 - positions
    gather_positions = torch.where(positions < lengths.unsqueeze(1), reversed_positions, positions).clamp_min(0)
    gather_positions = gather_positions.unsqueeze(-1).expand(batch_size, seq_len, dim)
    reversed_states = torch.gather(hidden_states, dim=1, index=gather_positions)
    return _apply_attention_mask(reversed_states, attention_mask)

def _restore_right_padded_sequence(hidden_states, attention_mask):
    if attention_mask is None:
        return torch.flip(hidden_states, dims=[1])
    return _reverse_right_padded_sequence(hidden_states, attention_mask)

@dataclass
class LncRFMConfig(PretrainedConfig):
    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    max_position_embeddings: int = 2048
    attn_layer_idx: list = (None,)
    attn_cfg: list = None
    position_embedding_type: str = 'absolute'
    use_bidirectional_mixer: bool = True
    pair_aware_fusion: bool = False
    bidirectional_strategy: str = 'mean'
    bidirectional_weight_tie: bool = True

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size=2, device=None, dtype=None, pad_token_id=0, layer_norm_eps=1e-12, hidden_dropout_prob=0.1, position_embedding_type='absolute'):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.position_embedding_type = position_embedding_type
        self.register_buffer('position_ids', torch.arange(max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)
        self.to(device=device, dtype=dtype)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values_length: int=0) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MaskedSelfAttentionMixer(nn.Module):

    def __init__(self, d_model, layer_idx=None, num_heads=12, dropout=0.0, bias=True, device=None, dtype=None, **kwargs):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'd_model={d_model} must be divisible by num_heads={num_heads}')
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, bias=bias, batch_first=True, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, attention_mask=None):
        hidden_states = _apply_attention_mask(hidden_states, attention_mask)
        key_padding_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            key_padding_mask = ~valid_mask
            all_padded = key_padding_mask.all(dim=1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded] = False
        output, _ = self.attn(hidden_states, hidden_states, hidden_states, key_padding_mask=key_padding_mask, need_weights=False)
        return _apply_attention_mask(output, attention_mask)

class BidirectionalMixer(nn.Module):

    def __init__(self, d_model, d_state=64, d_conv=7, conv_init=None, expand=2, headdim=64, ngroups=1, dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, dt_limit=(0.0, float('inf')), learnable_init_states=False, pair_aware_fusion=False, activation='swish', bias=False, conv_bias=True, chunk_size=256, use_mem_eff_path=False, layer_idx=None, device=None, dtype=None, **kwargs):
        super().__init__()
        if RMSNormGated is None:
            raise ImportError('mamba_ssm.ops.triton.layernorm_gated.RMSNorm is required for BidirectionalMixer')
        if use_mem_eff_path:
            raise ValueError('BidirectionalMixer in LncRFM uses the explicit scan path; set use_mem_eff_path=False')
        if activation not in {'silu', 'swish'}:
            raise ValueError(f'Unsupported bidirectional mixer activation={activation}')
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        if self.d_inner % self.headdim != 0:
            raise ValueError(f'expand*d_model={self.d_inner} must be divisible by headdim={self.headdim}')
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.pair_aware_fusion = bool(pair_aware_fusion)
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        d_in_proj = 2 * self.d_inner + 2 * (2 * self.ngroups * self.d_state) + 2 * self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        conv_dim = self.d_inner + 2 * (2 * self.ngroups * self.d_state)
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=conv_bias, kernel_size=d_conv, groups=conv_dim, padding=d_conv // 2, **factory_kwargs)
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True
        self.act = nn.SiLU()
        dt = torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        A = torch.ones(self.nheads, dtype=torch.float32, device=device)
        A_log = torch.log(A)
        if dtype is not None:
            A_log = A_log.to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)
        self.norm = RMSNormGated(self.d_inner, eps=1e-05, norm_before_gate=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.pair_fusion_gate = None
        if self.pair_aware_fusion:
            self.pair_fusion_gate = nn.Linear(1, 1, **factory_kwargs)
            nn.init.zeros_(self.pair_fusion_gate.weight)
            nn.init.zeros_(self.pair_fusion_gate.bias)

    def _fuse_bidirectional_outputs(self, y_fw, y_bw, structure_mask=None):
        if not self.pair_aware_fusion or structure_mask is None:
            return (y_fw + y_bw) / 2
        if structure_mask.size(1) > y_fw.size(1):
            structure_mask = structure_mask[:, :y_fw.size(1)]
        elif structure_mask.size(1) < y_fw.size(1):
            structure_mask = F.pad(structure_mask, (0, y_fw.size(1) - structure_mask.size(1)), value=0.0)
        pair_feature = structure_mask.to(device=y_fw.device, dtype=y_fw.dtype).unsqueeze(-1)
        gate = torch.sigmoid(self.pair_fusion_gate(pair_feature))
        return gate * y_fw + (1.0 - gate) * y_bw

    def forward(self, hidden_states, inference_params=None, attention_mask=None, seq_idx=None, structure_mask=None):
        hidden_states = _apply_attention_mask(hidden_states, attention_mask)
        batch = hidden_states.shape[0]
        zxbcdt = self.in_proj(hidden_states)
        A = -torch.exp(self.A_log.float())
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * (2 * self.ngroups * self.d_state), 2 * self.nheads], dim=-1)
        z = _apply_attention_mask(z, attention_mask)
        dt_fwd = dt[:, :, :self.nheads]
        dt_rev = _reverse_right_padded_sequence(dt[:, :, self.nheads:], attention_mask)
        dt = torch.cat((dt_fwd, dt_rev), dim=0)
        dt = F.softplus(dt + self.dt_bias)
        xBC = _apply_attention_mask(xBC, attention_mask)
        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        xBC = _apply_attention_mask(xBC, attention_mask)
        x, BC = torch.split(xBC, [self.d_inner, 2 * (2 * self.ngroups * self.d_state)], dim=-1)
        x_og = x
        x = torch.cat((x, _reverse_right_padded_sequence(x, attention_mask)), dim=0)
        branch_dim = 2 * self.ngroups * self.d_state
        BC = torch.cat((BC[:, :, :branch_dim], _reverse_right_padded_sequence(BC[:, :, branch_dim:], attention_mask)), dim=0)
        B, C = torch.split(BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = ssd_selective_scan(rearrange(x, 'b l (h p) -> b l h p', p=self.headdim), dt, A, rearrange(B, 'b l (g n) -> b l g n', g=self.ngroups), rearrange(C, 'b l (g n) -> b l g n', g=self.ngroups), D=None, z=None, dt_bias=None, dt_softplus=False, dt_limit=self.dt_limit)
        y = rearrange(y, 'b l h p -> b l (h p)')
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_fw = y[:batch]
        y_bw = _restore_right_padded_sequence(y[batch:], attention_mask)
        skip = x_og * repeat(F.linear(x_og, self.fc_D.weight, bias=self.D), 'b l h -> b l (h p)', p=self.headdim)
        y_fused = self._fuse_bidirectional_outputs(y_fw, y_bw, structure_mask)
        y = y_fused + skip
        y = self.norm(y, z)
        output = self.out_proj(y)
        return _apply_attention_mask(output, attention_mask)

def create_bidirectional_block(d_model, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, norm_epsilon=1e-05, rms_norm=False, residual_in_fp32=False, fused_add_norm=False, layer_idx=None, device=None, dtype=None, use_bidirectional_mixer=True, pair_aware_fusion=False, bidirectional_strategy='mean', bidirectional_weight_tie=True):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {'device': device, 'dtype': dtype}
    if layer_idx not in attn_layer_idx:
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        mixer_cls = partial(BidirectionalMixer, layer_idx=layer_idx, pair_aware_fusion=pair_aware_fusion, **ssm_cfg, **factory_kwargs)
    else:
        attn_cfg = copy.deepcopy(attn_cfg)
        attn_cfg.pop('use_masked_attention', False)
        mixer_cls = partial(MaskedSelfAttentionMixer, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    mlp_cls = nn.Identity
    block = Block(d_model, mixer_cls, mlp_cls=mlp_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block

class HybridEncoder(nn.Module):

    def __init__(self, d_model: int, n_layer: int, vocab_size: int, max_position_embeddings: int, ssm_cfg=None, norm_epsilon: float=1e-05, rms_norm: bool=False, initializer_cfg=None, fused_add_norm=False, residual_in_fp32=False, device=None, dtype=None, attn_layer_idx=None, attn_cfg=None, use_bidirectional_mixer=True, pair_aware_fusion=False, bidirectional_strategy='mean', bidirectional_weight_tie=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError('Failed to import Triton LayerNorm / RMSNorm kernels')
        self.layers = nn.ModuleList([create_bidirectional_block(d_model, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=i, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, use_bidirectional_mixer=use_bidirectional_mixer, pair_aware_fusion=pair_aware_fusion, bidirectional_strategy=bidirectional_strategy, bidirectional_weight_tie=bidirectional_weight_tie, **factory_kwargs) for i in range(n_layer)])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)
        self.apply(partial(_init_weights, n_layer=n_layer, **initializer_cfg if initializer_cfg is not None else {}))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=True, output_hidden_states=True, inference_params=None, structure_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        hidden_states = _apply_attention_mask(hidden_states, attention_mask)
        residual = None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            mixer_kwargs = {}
            if attention_mask is not None and isinstance(layer.mixer, (BidirectionalMixer, MaskedSelfAttentionMixer)):
                mixer_kwargs['attention_mask'] = attention_mask
            if structure_mask is not None and isinstance(layer.mixer, BidirectionalMixer):
                mixer_kwargs['structure_mask'] = structure_mask
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params, **mixer_kwargs)
            hidden_states = _apply_attention_mask(hidden_states, attention_mask)
            if residual is not None:
                residual = _apply_attention_mask(residual, attention_mask)
            current_attention = residual
            if output_attentions:
                all_attentions = all_attentions + (current_attention,)
        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(hidden_states, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps, residual=residual, prenorm=False, residual_in_fp32=self.residual_in_fp32)
        hidden_states = _apply_attention_mask(hidden_states, attention_mask)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs

class CLSPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        if first_token_tensor.dtype != self.dense.weight.dtype:
            first_token_tensor = first_token_tensor.to(self.dense.weight.dtype)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class LncRFMBackbone(nn.Module):

    def __init__(self, config: LncRFMConfig, initializer_cfg=None, device=None, dtype=None):
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        max_position_embeddings = config.max_position_embeddings
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        use_bidirectional_mixer = bool(getattr(config, 'use_bidirectional_mixer', False))
        pair_aware_fusion = getattr(config, 'pair_aware_fusion', False)
        bidirectional_strategy = getattr(config, 'bidirectional_strategy', 'mean')
        bidirectional_weight_tie = getattr(config, 'bidirectional_weight_tie', True)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.embedding = TokenEmbeddings(vocab_size=vocab_size, hidden_size=d_model, max_position_embeddings=max_position_embeddings, type_vocab_size=2, position_embedding_type=position_embedding_type, **factory_kwargs)
        self.pooler = CLSPooler(config).to(**factory_kwargs)
        self.encoder = HybridEncoder(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size, max_position_embeddings=max_position_embeddings, ssm_cfg=ssm_cfg, rms_norm=rms_norm, initializer_cfg=initializer_cfg, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, use_bidirectional_mixer=use_bidirectional_mixer, pair_aware_fusion=pair_aware_fusion, bidirectional_strategy=bidirectional_strategy, bidirectional_weight_tie=bidirectional_weight_tie, **factory_kwargs)
        self.apply(partial(_init_weights, n_layer=n_layer, **initializer_cfg if initializer_cfg is not None else {}))

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ids=None, labels=None, output_hidden_states=True, **kwargs):
        structure_mask = kwargs.get('structure_mask')
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Either input_ids or inputs_embeds must be provided.')
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('Only one of input_ids or inputs_embeds can be provided.')
        if input_ids is not None:
            B, L = input_ids.shape
            device = input_ids.device
        else:
            B, L, _ = inputs_embeds.shape
            device = inputs_embeds.device
        assert L <= self.config.max_position_embeddings, f'Input length {L} exceeds max_position_embeddings {self.config.max_position_embeddings}'
        if position_ids is None:
            position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embedding(input_ids=input_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(hidden_states, attention_mask, structure_mask=structure_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
