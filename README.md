# bs(block-sparse)-mask

The custom attention mechanisms can be found at:
https://github.com/emergenz/bs-mask/blob/912cac7f23884e949fda234be357713dcb7dbc14/src/custom_transformers/models/qwen2/modeling_qwen2.py#L226-L367

Note that I only adjusted the 'eager' attention, not the FlashAttention or SdpaAttention implementations. Therefore, I hardcoded using 'eager mode' at https://github.com/emergenz/bs-mask/blob/912cac7f23884e949fda234be357713dcb7dbc14/src/custom_transformers/models/qwen2/modeling_qwen2.py#L774

In `modeling_qwen2_no_causal.py`, I set `is_causal = False` at https://github.com/emergenz/bs-mask/blob/912cac7f23884e949fda234be357713dcb7dbc14/src/custom_transformers/models/qwen2/modeling_qwen2_no_causal.py#L227
