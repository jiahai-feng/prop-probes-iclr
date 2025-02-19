from functools import partial

import torch
import einops
from coref.expt_utils import pretty_print_logits, lookup_answer_logits
from coref.datasets.templates.common import stack_tokens, rotate
from coref.datasets.api import generate_tokenized_prompts
from coref.interventions import run_intervened_model
from transformer_lens.utilities.devices import get_device_for_block_index


class ScaleParameters:
    def __init__(self, parameters=None, scales={}):
        self.parameters = parameters
        self.scales = scales

    @staticmethod
    def default_adapter(parameters, layer):
        if layer in parameters.scales:
            return True, parameters.scales[layer] * parameters.parameters
        else:
            return False, None


class DictParameters:
    def __init__(self, /, model, layer_start, layer_end, n_dim=1):
        if model is None:
            self.parameters = {}
        else:
            self.parameters = {
                layer: torch.zeros(
                    (n_dim, model.cfg.d_model),
                    device=get_device_for_block_index(0, model.cfg),
                    requires_grad=True,
                    dtype=model.cfg.dtype,
                )
                for layer in range(layer_start, layer_end)
            }

    @classmethod
    def from_dict(cls, model, d):
        thing = cls(model=model, layer_start=0, layer_end=0)
        thing.parameters = d
        return thing

    def get_param_list(self):
        return list(self.parameters.values())

    @staticmethod
    def default_adapter(parameters, layer):
        if layer in parameters.parameters:
            return True, parameters.parameters[layer]
        else:
            return False, None

    @staticmethod
    def batch_adapter(parameters, layer, prompt_ids, prompt_id_start):
        """
        prompt_ids are slices of absolute values
        """
        assert prompt_ids is not None
        return (
            True,
            parameters.parameters[layer][
                slice(
                    prompt_ids.start - prompt_id_start,
                    prompt_ids.stop - prompt_id_start,
                )
            ],
        )

    @staticmethod
    def duplicate_adapter(parameters, layer, tag, start, end):
        if start <= layer and layer < end:
            return True, parameters.parameters[tag]
        return False, None

    @staticmethod
    def tag_adapter(parameters, layer, tag):
        return True, parameters.parameters[tag]

    @staticmethod
    def weight_loss(parameters, /, weight):
        return (
            weight
            * sum([v.norm() ** 2 for v in parameters.parameters.values()])
            / len(parameters.parameters)
        )


def general_inject_hook_fn(act, hook, *, param, width, token_maps, extractor):
    param = param.to(act.device)
    if len(param.shape) == 2:
        param = einops.repeat(param, "width dim -> batch width dim", batch=act.shape[0])
    else:
        assert param.shape[0] == act.shape[0]
        assert len(param.shape) == 3
    targets = extractor(token_maps)
    for batch in range(act.shape[0]):
        act[batch, targets[batch] : targets[batch] + width, :] += param[batch]
    return act


def general_overwrite_hook_fn(act, hook, param, width, token_maps, extractor):
    """
    param: [batch, width, dim]
    """
    param = param.to(act.device)
    if len(param.shape) == 2:
        param = einops.repeat(param, "width dim -> batch width dim", batch=act.shape[0])
    else:
        assert param.shape[0] == act.shape[0]
        assert len(param.shape) == 3
    targets = extractor(token_maps)
    assert targets.shape == (act.shape[0],)
    for batch in range(act.shape[0]):
        act[batch, targets[batch] : targets[batch] + width, :] = param[batch]
    return act


def project_switch_hook_fn(act, hook, *, param, width, token_maps, extractor):
    # extractor now returns two positions
    param = param.to(act.device)

    if len(param.shape) == 2:
        param = einops.repeat(param, "dim i -> w dim i", w=width)
    assert param.shape[1] == act.shape[2] and param.shape[0] == width

    targets1, targets2 = extractor(token_maps)
    assert targets1.shape == (act.shape[0],) and targets2.shape == (act.shape[0],)
    values = torch.zeros_like(act)
    for batch in range(act.shape[0]):
        proj1 = einops.einsum(
            param,
            param,
            act[batch, targets1[batch] : targets1[batch] + width, :],
            "w dim1 i, w dim2 i, w dim1 -> w dim2",
        )
        proj2 = einops.einsum(
            param,
            param,
            act[batch, targets2[batch] : targets2[batch] + width, :],
            "w dim1 i, w dim2 i, w dim1 -> w dim2",
        )
        values[batch, targets1[batch] : targets1[batch] + width, :] = proj2 - proj1
        values[batch, targets2[batch] : targets2[batch] + width, :] = proj1 - proj2
    act = act + values
    return act


def id_hook_fn(act, hook):
    return act


def build_example(
    *,
    batch_size,
    template,
    template_content,
    context_content,
    prompt_id_start,
    num_answers,
):
    """
    Args:

    Returns:
        all_tokens: (batch_size, length)
        all_answers: (batch_size, num_entities)
        all_token_maps: Recursive[IntTensor(batch_size, 2)] - 2 is for (start, end)
    """
    all_answers = []
    all_tokens = []
    all_token_maps = []
    for i in range(batch_size):
        prompt_id = i + prompt_id_start
        tokens, token_map, answers = generate_tokenized_prompts(
            template=template,
            template_content=template_content,
            context_content=context_content,
            prompt_id=prompt_id,
            num_answers=num_answers,
        )
        all_token_maps.append(token_map)
        all_answers.append(answers)
        all_tokens.append(tokens)
    all_answers = torch.tensor(all_answers)
    all_tokens = stack_tokens(template, all_tokens)
    all_token_maps = rotate(
        lambda substr_list: torch.tensor([[sub[0], sub[1]] for sub in substr_list])
    )(all_token_maps)
    return all_tokens, all_answers, all_token_maps


def interventions_to_hook_fns(interventions, prompt_ids, parsed_token_maps):
    '''
    interventions: Dict[layer: int, augmented_hook_fn: (act, hook, prompt_ids, token_maps) -> act]

    returns
        Dict[
            layer: int,
            hook_fn: (act, hook) -> act
        ]
    '''
    return {
        layer: partial(
            augmented_hook_fn,
            prompt_ids=prompt_ids,
            token_maps=parsed_token_maps,
        )
        for layer, augmented_hook_fn in interventions.items()
    }


def run_over_batches(
    *,
    model,
    num_samples,
    template,
    context_content,
    template_content,
    num_answers,
    prompt_id_start,
    batch_size=50,
    custom_runner,
):
    """
    custom_runner: (input_tokens, answer_tokens, stacked_token_maps, prompt_ids) -> Anything
    """
    all_rets = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        (source_tokens, answer_tokens, stacked_token_maps,) = build_example(
            batch_size=end - start,
            template=template,
            template_content=template_content,
            context_content=context_content,
            prompt_id_start=prompt_id_start + start,
            num_answers=num_answers,
        )
        prompt_ids = slice(prompt_id_start + start, prompt_id_start + end)
        intervened_logits = custom_runner(
            input_tokens=source_tokens,
            answer_tokens=answer_tokens,
            stacked_token_maps=stacked_token_maps,
            prompt_ids=prompt_ids,
        )
        all_rets.append(intervened_logits)
    return all_rets


def evaluate_interventions(
    *,
    model,
    num_samples,
    template,
    context_content,
    template_content,
    num_answers,
    prompt_id_start,
    interventions=None,
    canonize_token_maps=True,
    pos_deltas_fn=None,
    additional_answer_tokens=[],
    batch_size=50,
    return_raw=False,
    normalize_logits=True,
    return_full_logits=False,
    short_circuit=False,
    no_freezing=False,
    custom_runner=None,
):
    """
    Args:
        fixed_query_name : None | Name
        interventions: Dict[layer: int, augmented_hook_fn(tensor, hook, prompt_ids, token_maps)]
        pos_deltas_fn: (pos_deltas, parsed_token_maps) -> pos_deltas
    """
    answer_logits_sum = torch.zeros(num_answers + len(additional_answer_tokens))
    all_answer_logits = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        (source_tokens, answer_tokens, stacked_token_maps,) = build_example(
            batch_size=end - start,
            template=template,
            template_content=template_content,
            context_content=context_content,
            prompt_id_start=prompt_id_start + start,
            num_answers=num_answers,
        )
        prompt_ids = slice(prompt_id_start + start, prompt_id_start + end)

        if canonize_token_maps:
            parsed_token_maps = template.canonize_token_maps(stacked_token_maps)
        else:
            parsed_token_maps = stacked_token_maps
        # print(model.to_string(source_tokens)[0])
        # print(token_maps)
        # print(model.to_string(answer_tokens))
        # for batch in range(source_tokens.shape[0]):
        #     print(f'batch {batch}')
        #     print(', '.join(f'({vocab.tokenizer.decode(token)} {pos})' for pos, token in enumerate(source_tokens[batch])))

        if short_circuit:
            # no interventions, short circuit
            intervened_logits = model.get_logits(source_tokens)
        elif custom_runner is not None:
            intervened_logits = custom_runner(
                input_tokens=source_tokens,
                answer_tokens=answer_tokens,
                stacked_token_maps=stacked_token_maps,
                prompt_ids=prompt_ids,
            )
        else:
            intervened_logits = run_intervened_model(
                model=model,
                source_tokens=source_tokens,
                layer_hook_fns=interventions_to_hook_fns(
                    interventions, prompt_ids, parsed_token_maps
                ),
                dependent_start=0
                if no_freezing
                else stacked_token_maps["context_section"][:, 1],
                pos_deltas_fn=None
                if pos_deltas_fn is None
                else partial(pos_deltas_fn, token_maps=parsed_token_maps),
            )
        if additional_answer_tokens:
            additional_answer_tokens_tensor = torch.tensor(additional_answer_tokens).to(
                answer_tokens.device
            )
            answer_tokens = torch.cat(
                [
                    answer_tokens,
                    einops.repeat(
                        additional_answer_tokens_tensor,
                        "d-> batch d",
                        batch=answer_tokens.shape[0],
                    ),
                ],
                dim=1,
            )
        if not return_full_logits:
            answer_logits = lookup_answer_logits(
                intervened_logits,
                answer_tokens,
                query_position=stacked_token_maps["prompt"][:, 1] - 1,
                normalize=normalize_logits,
            )
        else:
            answer_logits = intervened_logits[:, -1, :]
        if not return_raw:
            answer_logits_sum += answer_logits.sum(dim=0).cpu()
        else:
            all_answer_logits.append(answer_logits.cpu())
    if not return_raw:
        return answer_logits_sum / num_samples
    else:
        return torch.cat(all_answer_logits, dim=0)


def build_sub_param(model, acts, category, side):
    """
    Constructs a DictParameters from activations obtained from collect_data
    acts should follow the structure:
        Dict[category:str -> List[ List[ Tensor[ batch, layer, dim]]]
    so that acts[category][side] gives a list of category_widths[category] tensors of size [batch, layer, dim]
    """
    dp = DictParameters.from_dict(
        model=model,
        d={
            layer: einops.rearrange(
                torch.stack([x[:, layer, :] for x in acts[category][side]]),
                "width batch dim -> batch width dim",
            )
            for layer in range(model.cfg.n_layers)
        },
    )
    return dp


def get_parallel_interventions(
    source_acts,
    category,
    source_side,
    target_side,
    category_widths,
    prompt_id_start,
    model,
):
    """
    Generate an intervention to substitute from source_acts
    Intervention has type:
        Dict[layer:int -> augmented_hook_fn]
    i.e., the same as the layer_interventions argument in evaluate_interventions
    """
    param = build_sub_param(model, source_acts, category, source_side)
    return {
        layer: lambda tensor, hook, prompt_ids, token_maps, layer=layer: general_overwrite_hook_fn(
            tensor,
            hook,
            param=param.batch_adapter(param, layer, prompt_ids, prompt_id_start)[1],
            width=category_widths[category],
            token_maps=token_maps,
            extractor=lambda token_maps: token_maps[category][target_side],
        )
        for layer in range(model.cfg.n_layers)
    }


def compose_augmented_hook_fns(*hook_fns):
    """ """

    def augmented_hook_fn(tensor, hook, prompt_ids, token_maps):
        for i in hook_fns:
            tensor = i(tensor, hook, prompt_ids=prompt_ids, token_maps=token_maps)
        return tensor

    return augmented_hook_fn


def compose_interventions(*interventions):
    ret = {}
    for li in interventions:
        for k, i in li.items():
            if k in ret:
                ret[k] = compose_augmented_hook_fns(ret[k], i)
            else:
                ret[k] = i
    return ret


def build_mean_param(model, la, lb):
    """
    la, lb: List[width -> Tensor[batch, layer, dim] ]
    Returns DictParameters that corresponds to la - lb
    """
    la = [a.mean(dim=0) for a in la]
    lb = [a.mean(dim=0) for a in lb]
    return DictParameters.from_dict(
        model=model,
        d={
            layer: torch.stack([(a - b)[layer, :] for a, b in zip(la, lb)])
            for layer in range(model.cfg.n_layers)
        },
    )


def get_inject_intervention(means, category, target_side, category_widths, model):
    """
    This only works if there are two statements
    """
    if target_side == 1:
        delta = DictParameters.from_dict(
            model=model, d={k: -v for k, v in means[category].parameters.items()}
        )
    else:
        delta = means[category]
    return {
        layer: lambda tensor, hook, prompt_ids, token_maps, layer=layer: general_inject_hook_fn(
            tensor,
            hook,
            param=delta.default_adapter(delta, layer)[1],
            width=category_widths[category],
            token_maps=token_maps,
            extractor=lambda token_maps: token_maps[category][target_side],
        )
        for layer in range(model.cfg.n_layers)
    }
