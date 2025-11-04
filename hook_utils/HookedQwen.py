from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from fancy_einsum import einsum
import einops
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    apply_rotary_pos_emb,
    repeat_kv,
    # ALL_ATTENTION_FUNCTIONS,
)
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa


class HookPoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = None

    def forward(self, x):
        return x


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]].to(
            attn_weights.device
        )
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    # value_states.shape: [batch, heads, seq, head_dim]
    value_states = module.hook_value_states_post_attn(value_states)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            # Goes here.
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    #breakpoint()
    #attn_output = scaled_dot_product_attention(
    #    query.float(),
    #    key.float(),
    #    value.float(),
    #    attn_mask=attention_mask,
    #    dropout_p=dropout,
    #    scale=scaling,
    #    is_causal=is_causal,
    #    **sdpa_kwargs,
    #)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
    #breakpoint()
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def hooked_forward_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    cos = cos.to(query_states.device)
    sin = sin.to(query_states.device)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        # attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_interface = sdpa_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        # sliding_window=self.sliding_window,  # diff with Llama
        is_causal=self.is_causal,
        **kwargs,
    )

    """
    # attn_weights: [batch, heads, seq (query), seq (key)]
    # attn_output: [batch, seq, heads, head_dim]
    attn_weights = self.hook_attn_pattern(attn_weights)
    # attn_output_reshaped: [batch, seq, d_model (heads * head_dim)]
    #attn_output_reshaped = attn_output.reshape(*input_shape, -1).contiguous()

    W_O = self.o_proj.weight #.clone()
    # [heads, d_head, d_model]
    W_O = einops.rearrange(W_O, "m (n h)->n h m", n=self.config.num_attention_heads)
    # self.o_proj: [d_model, d_model]
    #attn_output_final = self.hook_o_proj(self.o_proj(attn_output_reshaped))
    attn_output_per_head = einsum(
        "batch seq heads d_head, heads d_head d_model -> batch seq heads d_model",
        attn_output,
        W_O,
    )
    # [batch seq n_heads d_model]
    attn_output_per_head = self.hook_attn_out_per_head(attn_output_per_head)
    attn_output_final = attn_output_per_head.sum(dim=2)
    return attn_output_final, attn_weights
    """
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def hooked_forward_mlp(self, x):
    self.mlp_mid = self.hook_mlp_mid(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    down_proj = self.down_proj(self.mlp_mid)
    return down_proj


def hooked_forward_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs,
) -> torch.Tensor:

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states.to(residual.device)
    hidden_states = self.hook_resid_mid(hidden_states)

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states.to(residual.device)
    hidden_states = self.hook_resid_post(hidden_states)
    return hidden_states


def _convert_to_hooked_model(module):
    for child in module.children():

        if isinstance(child, Qwen3Attention):
            child.forward = hooked_forward_attention.__get__(child, Qwen3Attention)

        if isinstance(child, Qwen3MLP):
            child.forward = hooked_forward_mlp.__get__(child, Qwen3MLP)

        if isinstance(child, Qwen3DecoderLayer):
            child.forward = hooked_forward_decoder_layer.__get__(
                child, Qwen3DecoderLayer
            )

        _convert_to_hooked_model(child)


def convert_to_hooked_model_qwen(model):
    """
    This function sets up a hook for the model's forward pass.
    """
    for layer in model.model.layers:
        layer.hook_resid_mid = HookPoint()
        layer.hook_resid_post = HookPoint()

        layer.self_attn.hook_attn_pattern = HookPoint()
        layer.self_attn.hook_value_states_post_attn = HookPoint()
        layer.self_attn.hook_o_proj = HookPoint()
        layer.self_attn.hook_attn_out_per_head = HookPoint()

        layer.mlp.hook_mlp_mid = HookPoint()

    _convert_to_hooked_model(model)
