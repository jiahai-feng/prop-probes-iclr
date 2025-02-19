from functools import partial
import numpy as np
import torch
import einops
import torch.nn.functional as F

import coref.probes.activation_localization as act_loc
import coref.expt_utils as eu

import coref.parameters as p

from coref.datasets.domains.common import set_prompt_id
import coref.datasets.templates.common as tc
import coref.datasets.api as ta

from coref.probes.common import PredicateProbe
import coref.probes.centroid_probe as centroid_probe
import coref.form_processors
import logging
import os

class AffinityFns:
    @staticmethod
    def low_rank_affinity_fn(norm_acts, form, layer=15, dim=50):
        dtype = norm_acts.dtype
        subs = einops.einsum(
            norm_acts[:, layer].cuda(),
            ((form.U[:, :dim] * form.S[:dim]) @ form.Vh.T[:, :dim].T).to(dtype),
            norm_acts[:, layer].cuda(),
            "batch pos dim, dim dim2, batch pos2 dim2 -> batch pos pos2"
        )
        return subs
    @staticmethod
    def low_rank_inv_affinity_fn(norm_acts, form, layer=15, dim=50):
        dtype = norm_acts.dtype
        subs = einops.einsum(
            norm_acts[:, layer].cuda(),
            ((form.U[:, :dim] * form.S[:dim]) @ form.Vh.T[:, :dim].T).to(dtype).T, # we transpose the form
            norm_acts[:, layer].cuda(),
            "batch pos dim, dim dim2, batch pos2 dim2 -> batch pos pos2"
        )
        return subs
    
    @staticmethod
    def U_subspace_affinity_fn(norm_acts, form, layer=15, dim=50):
        dtype = norm_acts.dtype
        subs = einops.einsum(
            norm_acts[:, layer].cuda(),
            ((form.U[:, :dim] * form.S[:dim]) @ form.U[:, :dim].T).to(dtype),
            norm_acts[:, layer].cuda(),
            "batch pos dim, dim dim2, batch pos2 dim2 -> batch pos pos2"
        )
        return subs
    
    @staticmethod
    def U_subspace_sq_affinity_fn(norm_acts, form, layer=15, dim=50):
        dtype = norm_acts.dtype
        subs = einops.einsum(
            norm_acts[:, layer].cuda(),
            ((form.U[:, :dim] * form.S[:dim].pow(2)) @ form.U[:, :dim].T).to(dtype),
            norm_acts[:, layer].cuda(),
            "batch pos dim, dim dim2, batch pos2 dim2 -> batch pos pos2"
        )
        return subs
    
    @staticmethod
    def V_subspace_sq_affinity_fn(norm_acts, form, layer=15, dim=50):
        dtype = norm_acts.dtype
        subs = einops.einsum(
            norm_acts[:, layer].cuda(),
            ((form.Vh.T[:, :dim] * form.S[:dim].pow(2)) @ form.Vh.T[:, :dim].T).to(dtype),
            norm_acts[:, layer].cuda(),
            "batch pos dim, dim dim2, batch pos2 dim2 -> batch pos pos2"
        )
        return subs
    @staticmethod
    def V_subspace_affinity_fn(norm_acts, form, layer=15, dim=50):
        dtype = norm_acts.dtype
        subs = einops.einsum(
            norm_acts[:, layer].cuda(),
            ((form.Vh.T[:, :dim] * form.S[:dim]) @ form.Vh.T[:, :dim].T).to(dtype),
            norm_acts[:, layer].cuda(),
            "batch pos dim, dim dim2, batch pos2 dim2 -> batch pos pos2"
        )
        return subs
    
def get_affinity_fn(form_path, form_type, affinity_fn,  affinity_fn_kwargs):
    # convenience function
    # note that the probes here don't actually use this
    form = coref.form_processors.process_form(torch.load(form_path), form_type)
    return partial(
        getattr(AffinityFns, affinity_fn),
        form=form,
        **affinity_fn_kwargs
    )


def get_cache(model, template, c_slice, source_context, num_entities):
    source_tokens, answer_tokens, stacked_token_maps = p.build_example(
        batch_size=c_slice.stop - c_slice.start,
        template=template,
        template_content=dict(query_name=0),
        context_content=source_context,
        prompt_id_start=c_slice.start,
        num_answers=num_entities
    )
    _, source_cache = act_loc.get_normal_cache(
        model,
        input_tokens=source_tokens,
        prompt_ids=None, # not used
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
    )
    return source_cache, source_tokens
def cache_to_norm_acts(model, cache):
    acts = einops.rearrange(
        torch.stack([cache[f'blocks.{layer}.hook_resid_pre'].cpu() for layer in range(model.cfg.n_layers)]),
        'layer batch pos dim -> batch layer pos dim'
    )
    norm_acts = acts / acts.norm(dim=-1, keepdim=True)
    return norm_acts

from itertools import permutations

def bipartite_matching(scores):
    '''
    Maximum weight bipartite matching
    Args:
        scores: List[Tuple[Tuple[unique_value, other_value, other_type], score]]
    Returns:
        List[Tuple[unique_value, other_value, other_type]]
    '''

    # we just iterate over all possible matchings

    default_predicate = scores[0][0][2]
    left_values = sorted(list(set(pred[0] for pred, score in scores)))
    right_values = sorted(list(set(pred[1] for pred, score in scores)))
    edges = {
        (left, right): score
        for (left, right, _), score in scores
    }
    best_score = 0
    best_matching = None
    for assignments in permutations(range(len(right_values)), len(left_values)):
        # check assignment is valid
        has_invalid = False
        for left_id, right_id in enumerate(assignments):
            if (left_values[left_id], right_values[right_id]) not in edges:
                has_invalid = True
                break
        if has_invalid:
            continue
        # compute score
        score = sum(
            edges[(left_values[left_id], right_values[right_id])] 
            for left_id, right_id in enumerate(assignments)
        )
        if score > best_score:
            best_score = score
            best_matching = [
                (left_values[left_id], right_values[right_id], default_predicate)
                for left_id, right_id in enumerate(assignments)
            ]
    assert best_matching is not None
    return best_matching



def propose_predicates(
    norm_acts,
    index_probe,
    other_probes,
    binding_affinity_fn,
    context_mask,
    enforce_matching
):
    '''
    Args:
        norm_acts: Tensor[batch, layer, pos, dim]
        all_probes: List[DomainProbe]
        unique_probe: DomainProbe
        binding_affinity: (norm_acts: FloatTensor[batch, layer, pos, dim]) -> (affinity: FloatTensor[batch, pos, pos])
        enforce_matching: bool - if True, enforce 1-1 matching between names and attributes
    Returns:
        predicate_list: List[List[Tuple[unique_value, other_value, other_type]]
            - List of predicates for each batch
    Assumption: every attribute value is bound to some value of unique_probe
    '''
    num_batches, num_layers, num_pos, _ = norm_acts.shape
    binding_affinity = binding_affinity_fn(norm_acts)
    index_results = index_probe(norm_acts) # batch, pos
    other_results = [probe(norm_acts) for probe in other_probes]
    context_mask = context_mask.to(index_results.device)
    all_proposed_predicates = []
    for batch in range(num_batches):
        batch_predicates = []
        for probe, results in zip(other_probes, other_results):
            top_predicates = []
            all_predicates = []
            for pos in range(num_pos):
                if not context_mask[batch, pos]:
                    continue
                if results[batch, pos] != -1:
                    # find best unique_value using softmax
                    logits = binding_affinity[batch, pos, :]
                    mask = (index_results[batch, :] != -1) & context_mask[batch, :]
                    masked_logits = torch.where(mask, logits, -torch.inf)
                    assert len(masked_logits.shape) == 1
                    pattern = F.softmax(masked_logits, dim=0)
                    pattern = torch.nan_to_num(pattern, 0)
                    scores = np.zeros(len(index_probe.values))
                    for i, p in enumerate(pattern):
                        scores[index_results[batch, i].item()] += p
                    for i, s in enumerate(scores):
                        if s == 0:
                            continue
                        all_predicates.append((
                            (
                                index_probe.values[i],
                                probe.values[results[batch, pos].item()],
                                probe.type
                            ),
                            s
                        ))
                    
                    top_predicates.append((
                        index_probe.values[scores.argmax().item()],
                        probe.values[results[batch, pos].item()],
                        probe.type
                    ))
            if enforce_matching:
                # for each predicate, get the top score
                top_scores = {}
                for pred, score in all_predicates:
                    if pred not in top_scores or score > top_scores[pred]:
                        top_scores[pred] = score
                # do bipartite matching on the indices.
                batch_predicates.extend(bipartite_matching(list(top_scores.items())))
            else:
                batch_predicates.extend(top_predicates)
        all_proposed_predicates.append(batch_predicates)
    return all_proposed_predicates

class LookupProbe(PredicateProbe):
    def __init__(self, index_probe, other_probes, binding_affinity_fn, enforce_matching=False):
        self.index_probe = index_probe
        self.other_probes = other_probes
        self.binding_affinity_fn = binding_affinity_fn
        self.enforce_matching = enforce_matching
    def __call__(self, norm_acts, context_mask):
        return propose_predicates(
            norm_acts=norm_acts,
            index_probe=self.index_probe,
            other_probes=self.other_probes,
            binding_affinity_fn=self.binding_affinity_fn,
            context_mask=context_mask,
            enforce_matching=self.enforce_matching
        )

def compose_lookup_probe(
    model,
    use_cached_probes,
    probe_target_layer,
    form_path,
    das_path,
    form_type,
    affinity_fn,
    affinity_fn_kwargs,
    probe_cache_dir,
    has_country=True,
    has_food=True,
    has_occupation=False,
    chat_style='llama_chat',
    enforce_matching=False
):
    def get_probe(
        probe_class,
        save_name,
    ):
        probe = probe_class.load_or_none(probe_cache_dir / save_name)
        assert probe is not None, (f"Could not load probes from {probe_cache_dir / save_name}")
        return probe
    name_probe = get_probe(
        centroid_probe.NameCentroidProbe,
        'name_probe.pt'
    )
    other_probes = {}
    for enabled, probe_class, probe_name in [
        (
            has_country,
            centroid_probe.CountryCentroidProbe,
            'country_probe',
        ),
        (
            has_food,
            centroid_probe.FoodCentroidProbe,
            'food_probe',
        ),
        (
            has_occupation,
            centroid_probe.OccupationCentroidProbe,
            'occupation_probe',
        ),
    ]:
        if enabled:
            other_probes[probe_name] = get_probe(
                probe_class=probe_class,
                save_name=f"{probe_name}.pt"
            )
    form = coref.form_processors.get_form(
        form_type=form_type, 
        form_path=form_path, 
        das_path=das_path
    )
    lkp = LookupProbe(
        index_probe=name_probe,
        other_probes=list(other_probes.values()),
        binding_affinity_fn=partial(
            getattr(AffinityFns, affinity_fn),
            form=form,
            **affinity_fn_kwargs
        ),
        enforce_matching=enforce_matching
    )
    lkp.name_probe = name_probe
    for key, probe in other_probes.items():
        setattr(lkp, key, probe)
    return lkp

class NameCountryFoodProbe(LookupProbe):
    def __init__(
        self, 
        model,
        use_cached_probes,
        probe_target_layer,
        form_path,
        form_type,
        affinity_fn,
        affinity_fn_kwargs,
        probe_cache_dir,
    ):
        if use_cached_probes:
            name_probe = centroid_probe.NameCentroidProbe.load_or_none(probe_cache_dir / "name_probe.pt")
            country_probe = centroid_probe.CountryCentroidProbe.load_or_none(probe_cache_dir / "country_probe.pt")
            food_probe = centroid_probe.FoodCentroidProbe.load_or_none(probe_cache_dir / "food_probe.pt")
            if name_probe is None:
                logging.warn(f"Could not load probes from {probe_cache_dir}")
                name_probe, country_probe, food_probe = None, None, None
        else:
            name_probe, country_probe, food_probe = None, None, None
        if name_probe is None:
            os.makedirs(probe_cache_dir / "name_probe", exist_ok=True)
            os.makedirs(probe_cache_dir / "country_probe", exist_ok=True)
            os.makedirs(probe_cache_dir / "food_probe", exist_ok=True)

            name_probe = centroid_probe.NameCentroidProbe(target_layer=probe_target_layer)
            name_probe.train(model=model, save_dir=probe_cache_dir / "name_probe")
            country_probe = centroid_probe.CountryCentroidProbe(target_layer=probe_target_layer)
            country_probe.train(model=model, save_dir=probe_cache_dir / "country_probe")
            food_probe = centroid_probe.FoodCentroidProbe(target_layer=probe_target_layer)
            food_probe.train(model=model, save_dir=probe_cache_dir / "food_probe")

            name_probe.save(probe_cache_dir / "name_probe.pt")
            country_probe.save(probe_cache_dir / "country_probe.pt")
            food_probe.save(probe_cache_dir / "food_probe.pt")
        self.name_probe = name_probe
        self.country_probe = country_probe
        self.food_probe = food_probe
        form = coref.form_processors.process_form(torch.load(form_path), form_type)
        super().__init__(
            index_probe=name_probe,
            other_probes=[country_probe, food_probe],
            binding_affinity_fn=partial(
                getattr(AffinityFns, affinity_fn),
                form=form,
                **affinity_fn_kwargs
            )
        ) 

def get_norm_acts(chat_style, prefixes, contexts, model):
    '''
    Returns
        norm_acts : [batch, layer, pos, dim]
        mask : [batch, pos]
    '''
    input_tokens = []
    context_span = []
    template = BasicPrefixTemplate('llama')
    template.default_template_content['prompt_style'] = chat_style
    for prefix, context in zip(prefixes, contexts):
        tokens, token_maps, answers = ta.generate_tokenized_prompts(
            template=template,
            template_content=dict(query='', prefix=prefix, context=context),
            context_content=None,
            prompt_id=0,
            num_answers=0
        )
        input_tokens.append(tokens)
        context_span.append(token_maps['context'])
    input_tokens, pad_lengths = pad_right(input_tokens)

    _, source_cache = act_loc.get_normal_cache(
        model,
        input_tokens=input_tokens,
        prompt_ids=None, # not used
        stacked_token_maps=None,
        answer_tokens=None,
    )
    norm_acts = cache_to_norm_acts(model, source_cache)
    mask = torch.zeros_like(input_tokens, dtype=torch.bool)
    for i, (s, e) in enumerate(context_span):
        mask[i, s:e] = True
    return norm_acts, mask
class Decoder:
    def __init__(self, chat_style) -> None:
        self.chat_style = chat_style
    def __call__(self, prefixes, contexts, model):
        raise NotImplementedError
    def get_norm_acts(self, prefixes, contexts, model):
        '''
        Returns
            norm_acts : [batch, layer, pos, dim]
            mask : [batch, pos]
        '''
        return get_norm_acts(self.chat_style, prefixes, contexts, model)

from coref.datasets.templates.basic import BasicPrefixTemplate

def pad_right(input_tokens, pad_token_id=1):
    max_length = max(len(s) for s in input_tokens)
    padded = torch.tensor(
        [
            s + [pad_token_id] * (max_length - len(s))
            for s in input_tokens
        ]
    )
    pad_lengths = torch.tensor([max_length - len(s) for s in input_tokens])
    return padded, pad_lengths

class DomainProbeDecoder(Decoder):
    def __init__(self, probe, chat_style):
        self.probe = probe
        super().__init__(chat_style)
    def sweep_layers(self, prefixes, contexts, model):
        norm_acts, mask = self.get_norm_acts(prefixes, contexts, model)
        all_scores = self.probe.sweep_layers(norm_acts)
        return all_scores, mask
        
    def __call__(self, prefixes, contexts, model, output_scores=False):
        '''
        For every batch, returns a list of values detected by the probe.
        Each value is a tuple of (value, type)
        '''
        norm_acts, mask = self.get_norm_acts(prefixes, contexts, model)
        if output_scores:
            probe_results, probe_scores = self.probe(norm_acts, output_scores=True)
        else:
            probe_results = self.probe(norm_acts) # batch, pos
        all_values = centroid_probe.results_to_values(mask, probe_results, self.probe)
        if output_scores:
            return all_values, probe_scores, mask
        else:
            return all_values



class PredicateProbeDecoder(Decoder):
    def __init__(self, probe, chat_style):
        super().__init__(chat_style)
        self.probe = probe
    def __call__(self, prefixes, contexts, model):
        norm_acts, mask = self.get_norm_acts(prefixes, contexts, model)
        return self.probe(norm_acts, mask)
    
def evaluate_predicate_probe(
    model,
    dataset,
    decoder,
    num_samples,
    batch_size
):
    '''
    Dataset is a lists of dictionaries. Each dictionary must contain:
    - context
    - prompt_id
    - prefix
    - predicates
    '''
    all_predicted_predicates = []
    for c_slice in eu.generate_slices(0, num_samples, batch_size):
        predicted_predicates = decoder(
            prefixes=[d['prefix'] for d in dataset[c_slice]],
            contexts=[d['context'] for d in dataset[c_slice]],
            model=model
        )
        all_predicted_predicates.append(predicted_predicates)
    all_predicted_predicates = sum(all_predicted_predicates, [])
    all_true_predicates = [d['predicates'] for d in dataset]
    acc = [set(true_predicates) == set(predicates) for true_predicates, predicates in zip(all_true_predicates, all_predicted_predicates)]
    return sum(acc)/len(acc), all_predicted_predicates, all_true_predicates
        
def list_to_tuple(x):
    if isinstance(x, list):
        return tuple(list_to_tuple(i) for i in x)
    return x
def template_to_dataset(
    template,
    prompt_type,
    num_entities,
    num_samples,
    prompt_id_start
):
    dataset = []
    for prompt_id in range(prompt_id_start, prompt_id_start + num_samples):
        content_context = template.get_standard_context(num_entities)
        query_name = prompt_id % num_entities
        prompt = ta.generate_prompts(
            template=template,
            template_content=dict(
                query_name=query_name,
                prompt_type=prompt_type,
            ),
            context_content=content_context,
            prompt_id=prompt_id,
            num_answers=num_entities,
        )
        dataset.append({
            "context": prompt['prompt'][prompt['indices']['context_section'].to_slice()],
            "prompt_id": prompt_id,
            "prefix": "",
            "predicates": template.get_predicates(content_context, prompt_id)
        })
    return dataset

    
def evaluate_predicate_probe_templated(
    model,
    predicate_probe,
    num_samples,
    batch_size
):

    from coref.datasets.templates.fixed import NameCountryFoodFixedTemplate
    fixed_template = NameCountryFoodFixedTemplate('llama')
    dataset = template_to_dataset(
        template=fixed_template,
        prompt_type='chat_name_country',
        chat_style='llama_chat',
        num_entities=2,
        num_samples=num_samples,
        prompt_id_start=0
    )
    with open(logging.output_dir / "dataset.json", "w") as f:
        import json
        json.dump(dataset, f, indent=2)
    return evaluate_predicate_probe(
        model=model,
        dataset=dataset,
        decoder=PredicateProbeDecoder(predicate_probe),
        num_samples=num_samples,
        batch_size=batch_size
    )



class PromptDecoder(Decoder):
    stage1 = [
        "\n\nQuestion: How many people are mentioned in the above passage?",
        "The above passage mentions"
    ]
    number_map = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    stage2 = [
        "\n\nQuestion: What is the name of the {nth} person mentioned?",
        "The {nth} person mentioned is"
    ]
    nth_map = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    stage3 = {
        "food": [
            "\n\nQuestion: What food does {name} like?",
            "{name} likes to eat"
        ],
        "country": [
            "\n\nQuestion: What country is {name} from?",
            "{name} is from"
        ],
        "occupation": [
            "\n\nQuestion: What is the occupation of {name}?",
            "The occupation of {name} is"
        ]
    }
    def __init__(self, name_domain, country_domain, food_domain, occupation_domain=None, chat_style='llama_chat'):
        self.name_domain = name_domain
        self.country_domain = country_domain
        self.food_domain = food_domain
        self.occupation_domain = occupation_domain
        super().__init__(chat_style)
    def get_num_entities(self, prefixes, contexts, model):
        question, response = self.stage1
        template = BasicPrefixTemplate('llama')
        template.default_template_content['prompt_style'] = self.chat_style
        input_tokens = []
        last_tokens = []
        for prefix, context in zip(prefixes, contexts):
            tokens, token_maps, answers = ta.generate_tokenized_prompts(
                template=template,
                template_content=dict(query=response, prefix=prefix, context=context + question),
                context_content=None,
                prompt_id=0,
                num_answers=0
            )
            input_tokens.append(tokens)
            last_tokens.append(token_maps['prompt'].end - 1)
            assert token_maps['prompt'].end == len(tokens)
        input_tokens, pad_lengths = pad_right(input_tokens)
        logits = model(input_tokens)
        answer_logits = logits[
            torch.arange(len(prefixes))[:, None], 
            torch.tensor(last_tokens)[:, None],
            torch.tensor([model.tokenizer.encode(x)[1] for x in self.number_map])
        ]
        top_answer = answer_logits.argmax(dim=-1)
        return (top_answer + 1).cpu().tolist()

    def get_names(self, prefix, context, model, num_entities):
        question, response = self.stage2
        template = BasicPrefixTemplate('llama')
        template.default_template_content['prompt_style'] = self.chat_style
        input_tokens = []
        last_tokens = []
        for num in range(num_entities):
            tokens, token_maps, answers = ta.generate_tokenized_prompts(
                template=template,
                template_content=dict(
                    query=response.format(nth=self.nth_map[num]), 
                    prefix=prefix, 
                    context=context + question.format(nth=self.nth_map[num])
                ),
                context_content=None,
                prompt_id=0,
                num_answers=0
            )
            input_tokens.append(tokens)
            last_tokens.append(token_maps['prompt'].end - 1)
            assert token_maps['prompt'].end == len(tokens)
        input_tokens, pad_lengths = pad_right(input_tokens)
        logits = model(input_tokens)
        answer_logits = logits[
            torch.arange(num_entities)[:, None], 
            torch.tensor(last_tokens)[:, None],
            torch.tensor([self.name_domain.encode_single_word(v) for v in self.name_domain.data])
        ]
        top_answer = answer_logits.argmax(dim=-1)
        return [self.name_domain.data[i] for i in top_answer.cpu().tolist()]
    def get_food(self, prefix, context, model, names):
        domain = self.food_domain
        question, response = self.stage3['food']
        template = BasicPrefixTemplate('llama')
        template.default_template_content['prompt_style'] = self.chat_style

        input_tokens = []
        last_tokens = []
        for name in names:
            tokens, token_maps, answers = ta.generate_tokenized_prompts(
                template=template,
                template_content=dict(
                    query=response.format(name=name), 
                    prefix=prefix, 
                    context=context + question.format(name=name)
                ),
                context_content=None,
                prompt_id=0,
                num_answers=0
            )
            input_tokens.append(tokens)
            last_tokens.append(token_maps['prompt'].end - 1)
            assert token_maps['prompt'].end == len(tokens)
        input_tokens, pad_lengths = pad_right(input_tokens)
        logits = model(input_tokens)
        first_answer_tokens = torch.tensor([
            domain.tokenizer.encode(v)[1]
            for v in domain.data
        ])
        second_answer_tokens = torch.tensor([
            domain.tokenizer.encode(v)[2]
            for v in domain.data
        ])
        # print(domain.tokenizer.decode(first_answer_tokens))
        # print(domain.tokenizer.decode(second_answer_tokens))
        answer_logits = logits[
            torch.arange(len(names))[:, None], 
            torch.tensor(last_tokens)[:, None],
            first_answer_tokens
        ]
        first_answer = answer_logits.argmax(dim=-1) # batch
        input_tokens = torch.concatenate([
            input_tokens,
            torch.zeros((input_tokens.shape[0], 1), device=input_tokens.device, dtype=torch.long)
        ], dim=1)
        input_tokens[
            torch.arange(len(names), device=input_tokens.device), 
            torch.tensor(last_tokens, device=input_tokens.device) + 1
        ] = first_answer_tokens[first_answer.to(first_answer_tokens.device)].to(input_tokens.device)
        # print(domain.tokenizer.decode(first_answer_tokens[first_answer.to(first_answer_tokens.device)]))
        logits = model(input_tokens)
        answer_logits = logits[
            torch.arange(len(names))[:, None], 
            torch.tensor(last_tokens)[:, None] + 1,
            second_answer_tokens
        ] # batch answers
        candidate_answers = first_answer_tokens[None, :] == first_answer_tokens[first_answer.to(first_answer_tokens.device)][:, None]
        # print(candidate_answers)
        answer_logits = torch.where(candidate_answers.to(answer_logits.device), answer_logits, -torch.inf)
        final_answer = answer_logits.argmax(dim=-1)
        return [
            (name, domain.data[v], domain.type)
            for v, name in zip(final_answer.cpu().tolist(), names)
        ]
    def get_country(self, prefix, context, model, names):
        domain = self.country_domain
        question, response = self.stage3['country']
        template = BasicPrefixTemplate('llama')
        template.default_template_content['prompt_style'] = self.chat_style

        input_tokens = []
        last_tokens = []
        for name in names:
            tokens, token_maps, answers = ta.generate_tokenized_prompts(
                template=template,
                template_content=dict(
                    query=response.format(name=name), 
                    prefix=prefix, 
                    context=context + question.format(name=name)
                ),
                context_content=None,
                prompt_id=0,
                num_answers=0
            )
            input_tokens.append(tokens)
            last_tokens.append(token_maps['prompt'].end - 1)
            assert token_maps['prompt'].end == len(tokens)
        input_tokens, pad_lengths = pad_right(input_tokens)
        logits = model(input_tokens)
        answer_logits = logits[
            torch.arange(len(names))[:, None], 
            torch.tensor(last_tokens)[:, None],
            torch.tensor([
                domain.encode_single_word(v[0]) 
                for v in domain.data
            ])
        ]
        top_answer = answer_logits.argmax(dim=-1)
        return [
            (name, domain.data[v], domain.type)
            for v, name in zip(top_answer.cpu().tolist(), names)
        ]

    def get_occupation(self, prefix, context, model, names):
        domain = self.occupation_domain
        question, response = self.stage3['occupation']
        template = BasicPrefixTemplate('llama')
        template.default_template_content['prompt_style'] = self.chat_style
        

        input_tokens = []
        last_tokens = []
        for name in names:
            tokens, token_maps, answers = ta.generate_tokenized_prompts(
                template=template,
                template_content=dict(
                    query=response.format(name=name), 
                    prefix=prefix, 
                    context=context + question.format(name=name)
                ),
                context_content=None,
                prompt_id=0,
                num_answers=0
            )
            input_tokens.append(tokens)
            last_tokens.append(token_maps['prompt'].end - 1)
            assert token_maps['prompt'].end == len(tokens)
        input_tokens, pad_lengths = pad_right(input_tokens)
        logits = model(input_tokens)
        answer_logits = logits[
            torch.arange(len(names))[:, None], 
            torch.tensor(last_tokens)[:, None],
            torch.tensor([
                domain.encode_single_word(v) 
                for v in domain.data
            ])
        ]
        top_answer = answer_logits.argmax(dim=-1)

        occ = [
            (name, domain.data[v], domain.type)
            for v, name in zip(top_answer.cpu().tolist(), names)
        ]
        # print(f'Occupations for {names}: {occ}')
        # print(f'Context: {context}')
        return occ
    def __call__(self, prefixes, contexts, model):
        '''
        3 stage prompting pipeline
        1) How many people are mentioned in?
        2) What is the name of the first person mentioned?
        3) What food/country etc. does X like?

        Args:
            prefixes: List[str]
            contexts: List[str]
            model
        '''
        all_num_entities = self.get_num_entities(prefixes, contexts, model)
        # print(f'Number of entities: {all_num_entities}')
        all_names = [
            self.get_names(prefix, context, model, num)
            for prefix, context, num in zip(prefixes, contexts, all_num_entities)
        ]
        # print(f'Names: {all_names}')
        all_predicates = []
        all_predicates.append([
            self.get_country(prefix, context, model, names)
            for prefix, context, names in zip(prefixes, contexts, all_names)
        ])
        if self.food_domain is not None:
            all_predicates.append([
                self.get_food(prefix, context, model, names)
                for prefix, context, names in zip(prefixes, contexts, all_names)
            ])
        if self.occupation_domain is not None:
            all_predicates.append([
                self.get_occupation(prefix, context, model, names)
                for prefix, context, names in zip(prefixes, contexts, all_names)
            ])
        return [
            sum(predicate_lists, []) for predicate_lists in zip(*all_predicates)
        ]
