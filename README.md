# Monitoring Language Models with Propositional Probes

[arxiv](https://arxiv.org/abs/2406.19501)

This contains an implementation of propositional probes as described in the arxiv paper.

Python Environment
---
1. Get [uv](https://docs.astral.sh/uv/)
2. Run `uv sync`
3. Activate the environment `source .venv/bin/activate`, or `uv run` everything.


Environment variables
---
Populate `.env` at root folder with:
```
HF_TOKEN=<hf token (for llama)>
OPENAI_API_KEY=<open ai key>
HF_HOME=<cache directory for HF>
```


Datasets
---
To download the winobias dataset, tun the following from the project root directory
```
mkdir -p src/coref/datasets/raw/winobias/
wget -P src/coref/datasets/raw/winobias/ https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/anti_stereotyped_type1.txt.dev
wget -P src/coref/datasets/raw/winobias/ https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/pro_stereotyped_type1.txt.dev
wget -P src/coref/datasets/raw/winobias/ https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/anti_stereotyped_type2.txt.dev
wget -P src/coref/datasets/raw/winobias/ https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/pro_stereotyped_type2.txt.dev


wget -P src/coref/datasets/raw/winobias/ https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/female_occupations.txt
wget -P src/coref/datasets/raw/winobias/ https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/male_occupations.txt
```

## Synthetic datasets
The synthetic datasets are already in the repo under `exports/datasets`.

To replicate their creation, run the following scripts:

Export synthetic dataset (name-country-food-occupation)
```
python -m scripts.export_dataset --template "NameCountryFoodOccupationFixedTemplate" \
    --prompt_type 'chat_name_country' --context_type 'basic' \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_food_occupation_basic
```

Generate paraphrases (name-country-food-occupation)
```
python -m coref.datasets.auto_gen.process_prompt \
    --prompt_path exports/datasets/name_country_food_occupation_basic/dataset.json \
    --output_dir exports/datasets/name_country_food_occupation_basic/paraphrase/ \
    --gpt_instruct_path coref/datasets/auto_gen/rewrite_prompts/paraphrase.txt
```

Generate translations (name-country-food-occupation)
```
python -m coref.datasets.auto_gen.process_prompt \
    --prompt_path exports/datasets/name_country_food_occupation_basic/paraphrase/dataset.json \
    --output_dir exports/datasets//name_country_food_occupation_basic/es_translation/ \
    --gpt_instruct_path coref/datasets/auto_gen/rewrite_prompts/translate_to_es.txt \
    --prompt_header "context"
```

Validation set (name-country-food-occupation)
```
python -m scripts.export_dataset --template "NameCountryFoodOccupationFixedTemplate" \
    --prompt_type 'chat_name_country' --context_type 'basic' \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --prompt_id_start 512 \
    --output_dir exports/datasets/name_country_food_occupation_basic_val

python -m coref.datasets.auto_gen.process_prompt \
    --prompt_path exports/datasets/name_country_food_occupation_basic_val/dataset.json \
    --output_dir exports/datasets/name_country_food_occupation_basic_val/paraphrase/ \
    --gpt_instruct_path coref/datasets/auto_gen/rewrite_prompts/paraphrase.txt

python -m coref.datasets.auto_gen.process_prompt \
    --prompt_path exports/datasets/name_country_food_occupation_basic_val/paraphrase/dataset.json \
    --output_dir exports/datasets/name_country_food_occupation_basic_val/es_translation/ \
    --gpt_instruct_path coref/datasets/auto_gen/rewrite_prompts/translate_to_es.txt \
    --prompt_header "context"
```

Additionally, for the systematic order analyses in Appendix I, generate the following:

```
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'series' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_series_basic
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'cross' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_cross_basic
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'nested' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_nested_basic
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'medium' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_medium_basic
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'long' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_long_basic
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'nested_2' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_nested_2_basic
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'coref' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_coref_basic
python -m scripts.export_dataset --template "NameCountryVaryFixedTemplate" \
    --context_type 'reverse' --split_sep True \
    --chat_style 'sep' --num_entities 2 --num_samples 512 \
    --output_dir exports/datasets/name_country_reverse_basic
```

Finally, to finetune the unfaithful spanish model, we need to create training data.
```
python -m scripts.export_finetuning_dataset
```

Setting up an artifact directory
---
We provide artifacts for the propositional probes. Specifically, we provide the weights of the domain probes and the Hessian binding subspaces. Download and unzip this file. We will call `.../prop-probes-artifacts` the artifact directory.

If you wish to train these artifacts yourself, you can follow instructions below. You should still prepare an empty directory as the artifact directory.


Hessian method for Binding Subspace
---
You can replicate the Hessian binding subspace using `notebooks/subspace_experiments.ipynb`. 

- The notebook contains slurm commands for running experiments. 
- The notebook also generates the necessary config files to run the experiments --- you should thus at least run those cells even if you don't use slurm. 
- To run the notebook, you have to first specify the output and artifact directory.

Broadly speaking, `scripts/run_hessians.py` is the main entrypoint for running the Hessian experiments. For example, to run Hessians on tulu:
```
python -m scripts.run_hessians \
    --config experiments/point_hessians/paper/tulu_scale_False_interpolating_0.5.yaml
```
The binding subspace in our provided artifacts is the output files of the `run_hessians` commands in the notebook.


Constructing and evaluating Propositional Probes
---
Refer to `notebooks/probe_experiments.ipynb` for slurm commands for running experiments.

- To run the notebook, you have to first specify the output and artifact directory.
- To run experiments involving dataset poisoning, you'd have to first finetune them and save the unfaithful models in the appropriate artifact directory! (see below section)

Finetuning unfaithful model
---
Set the artifact and `coref_root` directories in `experiments/finetune/llama_finetune.yaml` and `experiments/finetune/tulu_finetune.yaml`

`coref_root` should be the path to the root of this repo.

```
python -m coref.finetune --config experiments/finetune/tulu_finetune.yaml
python -m coref.finetune --config experiments/finetune/llama_finetune.yaml
```

Plotting
---
See `notebooks/paper_plots.ipynb`.

