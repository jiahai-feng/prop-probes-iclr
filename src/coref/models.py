import torch
import re
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import transformer_lens.HookedTransformer as HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name
import os
from functools import cache


LLAMA_PATH = os.environ["LLAMA_WEIGHTS"] if "LLAMA_WEIGHTS" in os.environ else None
HF_PATH = os.environ["HF_CACHE"] if "HF_CACHE" in os.environ else None
TULU_PATH = os.environ["TULU_WEIGHTS"] if "TULU_WEIGHTS" in os.environ else None

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]
LLAMA_MODELS = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]
TULU_MODELS = [
    "tulu-7b",
    "tulu-13b",
    "tulu-30b",
    "tulu-65b",
]
TULU_2_MODELS = ["tulu-2-13b", "tulu-2-7b", "codetulu-2-13b", "codetulu-2-7b"]

LLAMA_2_MODELS = [
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Llama-2-70b-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
]


def get_base_llama_from_tulu(model_tag):
    model_size = re.search(r"(\d+)b", model_tag).group(1)
    llama_tag = f"llama-{model_size}b-hf"
    return llama_tag

hf_model_map = {
    **{model: f'meta-llama/{model}' for model in LLAMA_2_MODELS},
    **{model: f'allenai/{model}' for model in TULU_2_MODELS},
}
base_model_map = {
    **{model: model for model in LLAMA_2_MODELS},
    **{model: get_base_llama_from_tulu(model) for model in TULU_2_MODELS},
}

def get_hf_model(model_tag, peft_dir, local_dir, dtype):
    assert not (peft_dir is not None and local_dir is not None), "Cannot specify both peft_dir and local_dir"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_map[model_tag] if local_dir is None else local_dir,
        **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if peft_dir is not None:
        peft_model = PeftModel.from_pretrained(hf_model, peft_dir)
        hf_model = peft_model.merge_and_unload()
    return hf_model


def fetch_model(model_tag, device="cuda", num_devices=1, dtype=torch.float32, hf=False, peft_dir=None, local_dir=None):
    torch.set_grad_enabled(False)
    if hf:
        if model_tag in hf_model_map:
            hf_model = get_hf_model(model_tag, peft_dir, local_dir, dtype)
            hf_model.tokenizer = get_llama_tokenizer()

            class cfg:
                device = "cuda"
                n_layers = len(hf_model.model.layers)
                d_model = hf_model.model.embed_tokens.weight.shape[1]

            hf_model.cfg = cfg
            hf_model.get_logits = lambda input_ids: hf_model(input_ids).logits
        else:
            raise f"Unrecognized HF model {model_tag}"
        return hf_model

    if model_tag in PYTHIA_MODELS:
        hf_model = AutoModelForCausalLM.from_pretrained(
            get_official_model_name(model_tag),
            **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        model = HookedTransformer.from_pretrained_no_processing(
            model_tag,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            hf_model=hf_model,
            dtype=dtype,
        )
    elif model_tag in LLAMA_MODELS:
        LLAMA_SIZE = re.match(r"llama-(\d+)b-hf", model_tag).group(1)
        MODEL_PATH = os.path.join(LLAMA_PATH, f"{LLAMA_SIZE}B")
        tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_PATH)
        hf_model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=dtype
        )
        model = HookedTransformer.from_pretrained_no_processing(
            model_tag,
            hf_model=hf_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype,
        )
        model.tokenizer = tokenizer
        model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif model_tag in TULU_MODELS:
        model_size = re.match(r"tulu-(\d+)b", model_tag).group(1)
        llama_tag = f"llama-{model_size}b-hf"
        model_path = os.path.join(TULU_PATH, f"tulu-merged-{model_size}b")
        # probably okay to use llama tokenizer instead of tulu tokenizer
        # should be just a difference of a padding token at the end
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        hf_model = LlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=dtype
        )
        model = HookedTransformer.from_pretrained_no_processing(
            llama_tag,
            hf_model=hf_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype,
        )
        model.tokenizer = tokenizer
        # the new model already has this i think
        # model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif model_tag in hf_model_map:
        hf_model = get_hf_model(model_tag, peft_dir, local_dir, dtype)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = HookedTransformer.from_pretrained_no_processing(
            base_model_map[model_tag],
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            hf_model=hf_model,
            dtype=dtype,
        )
        model.tokenizer = tokenizer
    else:
        raise Exception(f"Unknown model tag {model_tag}")
    model.eval()
    model.model_tag = model_tag
    model.get_logits = lambda input_ids: model(input_ids)
    return model


@cache
def get_llama_tokenizer():
    if LLAMA_PATH is not None:
        return LlamaTokenizerFast.from_pretrained(os.path.join(LLAMA_PATH, "7B"))
    else:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


@cache
def get_pythia_tokenizer():
    return AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-2.8b",
        **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
    )

def get_tokenizer(tokenizer_type):
    if tokenizer_type == "pythia":
        return get_pythia_tokenizer()
    elif tokenizer_type == "llama":
        return get_llama_tokenizer()
    else:
        raise Exception(f"Unknown tokenizer type {tokenizer_type}")
