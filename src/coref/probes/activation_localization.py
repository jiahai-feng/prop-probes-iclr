import torch
import coref.interventions as cint
import coref.saliency as sal
import einops
'''
similar to sal, but this is on resid_pre and not on the attn z
'''


def get_normal_cache(model, input_tokens, prompt_ids, stacked_token_maps, answer_tokens):
    logits, normal_cache = cint.run_cached_model( 
        execution_fn=lambda: model.get_logits(input_tokens),
        model=model,
        names_filter=lambda x: "hook_resid_pre" in x,
        incl_bwd=False,
    )
    return logits.detach(), normal_cache
def switch_run(
    model, input_tokens, prompt_ids, stacked_token_maps, answer_tokens, source_cache
):
    def patch_resid_pre(act, hook):
        mask = torch.zeros_like(act, dtype=torch.bool)
        for i in range(input_tokens.shape[0]):
            s, e = stacked_token_maps['context_section'][i, :]
            mask[i, s:e, :] = True
        act = torch.where(
            mask,
            source_cache[hook.name][prompt_ids],
            act
        )
        return act
    return cint.run_intervened_model(
        model=model,
        source_tokens=input_tokens,
        layer_hook_fns={
            layer: patch_resid_pre
            for layer in range(model.cfg.n_layers)
        },
        dependent_start=stacked_token_maps["context_section"][:, 1],
        pos_deltas_fn=None,
    )
    
def get_abnormal_cache(
    model,
    input_tokens, prompt_ids, stacked_token_maps, answer_tokens, hook_context, abnormal_run
):
    cache = {}

    def cache_forward(tensor, hook):
        cache[hook.name] = tensor

    def cache_backward(tensor, hook):
        cache[hook.name + "_grad"] = tensor

    def block_qk_hook(grad, hook):
        if hook_context["block_qk"]:
            # we're blocking QK gradients entirely
            # in saliency, we allowed QK to flow if it came from the context acts
            ret = torch.zeros_like(grad)
            return (ret,)

    def block_attn_hook(grad, hook):
        if hook_context["block_attn"]:
            ret = torch.zeros_like(grad)
            return (ret,)

    with torch.enable_grad():
        with model.hooks(
            fwd_hooks=[
                (f"blocks.{layer}.hook_resid_pre", cache_forward)
                for layer in range(model.cfg.n_layers)
            ],
            bwd_hooks=(
                [
                    (f"blocks.{layer}.hook_resid_pre", cache_backward)
                    for layer in range(model.cfg.n_layers)
                ]
                + [
                    (f"blocks.{layer}.attn.hook_pattern", block_qk_hook)
                    for layer in range(model.cfg.n_layers)
                ]
                + [
                    (f"blocks.{layer}.attn.hook_z", block_attn_hook)
                    for layer in range(model.cfg.n_layers)
                ]
            ),
        ):
            logits = abnormal_run(
                model=model,
                input_tokens=input_tokens,
                prompt_ids=prompt_ids,
                stacked_token_maps=stacked_token_maps,
                answer_tokens=answer_tokens,
            )
    return logits, cache


def discover_tokens(
    model,
    input_tokens,
    prompt_ids,
    stacked_token_maps,
    answer_tokens,
    answer,
    abnormal_run,
    block_qk=True,
    loss="logit_diff",
):
    """
    Returns:
        total_score: FloatTensor[pos, layer]
        abnormal_cache: Dict[str, Tensor]
    abnormal_cache contains:
        "blocks.{layer}.hook_resid_pre": Tensor[batch, pos, d_model]
        and
        "blocks.{layer}.hook_resid_pre_grad": Tensor[batch, pos, d_model]
    """
    sal.dim_model_grads(model)
    hook_context = {}
    normal_logits, normal_cache = get_normal_cache(
        model,
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
    )
    logits, abnormal_cache = get_abnormal_cache(
        model,
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
        hook_context=hook_context,
        abnormal_run=abnormal_run,
    )
    normal_log_prob = normal_logits[
        torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]
    ] - torch.logsumexp(logits[:, -1, :], dim=1, keepdim=False)
    abnormal_log_prob = logits[
        torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]
    ] - torch.logsumexp(logits[:, -1, :], dim=1, keepdim=False)
    print(f'Normal log_prob: {normal_log_prob.mean().item()}')
    print(f'Abnormal log_prob: {abnormal_log_prob.mean().item()}')

    forward_name = lambda layer: f"blocks.{layer}.hook_resid_pre"
    backward_name = lambda layer: f"blocks.{layer}.hook_resid_pre_grad"

    # relax root without blocking attn

    hook_context["block_attn"] = False
    hook_context["block_qk"] = block_qk
    sal.wipe_backwards(abnormal_cache)

    with torch.enable_grad():
        if loss == "logit_diff":
            logit_diff = (
                logits[torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]]
                - logits[
                    torch.arange(logits.shape[0]), -1, answer_tokens[:, 1 - answer]
                ]
            )
            assert logit_diff.shape == (logits.shape[0],)
            logit_diff.mean().backward(retain_graph=True)
        elif loss == "log_prob":
            log_prob = logits[
                torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]
            ] - torch.logsumexp(logits[:, -1, :], dim=1, keepdim=False)
            assert log_prob.shape == (logits.shape[0],)
            log_prob.mean().backward(retain_graph=True)
        elif loss == 'logit':
            logit = logits[torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]]
            logit.mean().backward(retain_graph=True)
        else:
            assert False, f"Unknown loss type {loss}"
    with torch.no_grad():
        total_score = einops.rearrange(
            torch.stack(
                [
                    einops.einsum(
                        abnormal_cache[forward_name(layer)].detach()
                        - normal_cache[forward_name(layer)],
                        abnormal_cache[backward_name(layer)].detach()
                        if backward_name(layer) in abnormal_cache
                        else torch.zeros_like(abnormal_cache[forward_name(layer)]),
                        "batch pos d_model, batch pos d_model -> batch pos",
                    ).cpu()
                    for layer in range(model.cfg.n_layers)
                ]
            ),
            "layer batch pos -> batch pos layer",
        ).sum(dim=0)
    # Kill backward graph
    with torch.enable_grad():
        logits.mean().backward()

    return total_score, abnormal_cache