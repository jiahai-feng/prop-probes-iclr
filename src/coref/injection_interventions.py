import torch
import coref.parameters as p


def paramify_acts(model, acts):
    act_means = [a.mean(dim=0) for a in acts]
    return p.DictParameters.from_dict(
        model=None,
        d={
            layer: torch.stack([a[layer, :] for a in act_means])
            for layer in range(model.cfg.n_layers)
        },
    )


def injection_intervention_from_param(
    model, param, width, category, target_side, adapter
):
    return {
        layer: lambda tensor, hook, prompt_ids, token_maps, layer=layer: p.general_inject_hook_fn(
            tensor,
            hook,
            param=adapter(param, layer)[1],
            width=width,
            token_maps=token_maps,
            extractor=lambda token_maps: token_maps[category][target_side],
        )
        for layer in range(model.cfg.n_layers)
        if adapter(param, layer)[0]
    }


def negate_adapter(adapter):
    def ret(param, layer):
        x, y = adapter(param, layer)
        if x:
            return x, -y
        else:
            return x, y

    return ret


def scale_adapter(adapter, scale):
    def ret(param, layer):
        x, y = adapter(param, layer)
        if x:
            return x, scale * y
        else:
            return x, y

    return ret


def build_switch_interventions(
    model, category_widths, param, category, target_sides=[2, 3]
):
    return [
        injection_intervention_from_param(
            model=model,
            param=param,
            width=category_widths[category],
            category=category,
            target_side=target_sides[0],
            adapter=param.default_adapter,
        ),
        injection_intervention_from_param(
            model=model,
            param=param,
            width=category_widths[category],
            category=category,
            target_side=target_sides[1],
            adapter=negate_adapter(param.default_adapter),
        ),
    ]


def build_add_interventions(model, category_widths, params, category, target_sides):
    return [
        injection_intervention_from_param(
            model=model,
            param=param,
            width=category_widths[category],
            category=category,
            target_side=side,
            adapter=param.default_adapter,
        )
        for param, side in zip(params, target_sides)
    ]
