"""Custom eager attention with prediction slot isolation.

The R copies within a ``<value>`` / ``<relative_value>`` slot attend
bidirectionally to each other; cross-slot pred keys are invisible to all
other queries.

All extra tensors flow through HF's ``**kwargs`` pass-through from the
top-level ``forward()`` call down to the attention function.
"""

from typing import Optional

import torch
import torch.nn as nn

from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def pred_slot_isolated_eager(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    is_pred_key: Optional[torch.Tensor] = None,
    pred_slot_id: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B = query.shape[0]
    L_q = query.shape[2]
    L_k = key_states.shape[2]
    mask_dtype = query.dtype
    neg_inf = torch.finfo(mask_dtype).min

    if attention_mask is not None:
        add_mask = attention_mask[:, :, :, :L_k].to(mask_dtype).clone()
    else:
        add_mask = torch.zeros(B, 1, L_q, L_k, dtype=mask_dtype, device=query.device)

    if is_pred_key is not None and is_pred_key.any():
        if pred_slot_id is not None:
            slot_q = pred_slot_id[:, :, None]
            slot_k = pred_slot_id[:, None, :]
            same_slot = (slot_q == slot_k) & (slot_q >= 0)
            add_mask = add_mask.masked_fill(same_slot[:, None, :, :], 0.0)
            pred_mask = is_pred_key[:, None, :] & ~same_slot
        else:
            # Decode step with KV cache: the new query is never a pred token,
            # so every cached pred key must be masked.
            pred_mask = is_pred_key[:, None, :].expand(B, L_q, L_k)
        add_mask = add_mask.masked_fill(pred_mask[:, None, :, :], neg_inf)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling + add_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


ALL_ATTENTION_FUNCTIONS["pred_slot_isolated_eager"] = pred_slot_isolated_eager
ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["pred_slot_isolated_eager"] = eager_mask
