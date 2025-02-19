Log of edits:

- Adds hacky patch to `HookedTransformer.from_pretrained` to allow for Tulu
- Slight modification to `get_device_for_block_index` for better rebalancing
- Moves `attention_mask` to correct device in `HookedTransformer.forward`
- Changed changed register_full_backward__hook to register_full_backward_pre_hook in `hook_points`.