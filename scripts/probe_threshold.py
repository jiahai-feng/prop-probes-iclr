import torch
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from coref import COREF_ROOT
import coref.run_manager as rm

import coref.probes.centroid_probe as centroid_probe
from coref.probes.evaluate import recompute_domain_values


def get_last_output(cfg_path):
    parent_dir = Path(rm.get_run_dir_parent(cfg_path, outputs_root, expts_root))
    dirs = [d for d in os.listdir(parent_dir)  if os.path.isdir(parent_dir / d)]
    success_dir = [d for d in dirs if 'done.out' in os.listdir(parent_dir / d)]
    max_run = max(int(d) for d in dirs)
    max_success = max(int(d) for d in success_dir)
    if max_run != max_success:
        print(f'Warning: latest run {max_run} of {cfg_path} is not successful. Falling back to {max_success}')
    return parent_dir / str(max_success)
        

def get_probe_output_dir(dataset, model, use_class_conditioned):
    cfg_path = expts_root / f'probes/eval_domain/{dataset}_{model}_use_class_{use_class_conditioned}.yaml'
    output_dir = get_last_output(cfg_path)
    return output_dir

def get_probe_save_dir(dataset, model, use_class_conditioned):
    cfg_path = expts_root / f'probes/eval_domain/{dataset}_{model}_use_class_{use_class_conditioned}.yaml'
    cfg, meta_kwargs = rm.load_cfg(cfg_path)
    return cfg['probe_cache_dir']

def accuracy_sweep(dataset, model, use_class_conditioned):
    all_accuracies = []
    params = [
        ('name_probe', centroid_probe.NameCentroidProbe),
        ('country_probe', centroid_probe.CountryCentroidProbe),
        ('occupation_probe', centroid_probe.OccupationCentroidProbe),
        ('food_probe', centroid_probe.FoodCentroidProbe),
    ]
    for probe_name, probe_class in params:
        details_path = get_probe_output_dir(dataset, model, use_class_conditioned)/probe_name/'details.pt'
        details = torch.load(details_path)

        all_scores = details['all_scores'] # values, batch, pos
        all_masks = details['all_masks']
        all_true_values = details['all_true_values']
        probe_dir = Path(get_probe_save_dir(dataset, model, use_class_conditioned)) / f'{probe_name}.pt'

        probe = probe_class.load_or_none(probe_dir)


        def get_accuracy(threshold):
            all_predicted_values = recompute_domain_values(
                mask=all_masks,
                scores=all_scores,
                probe=probe,
                threshold=threshold
            )
            return np.mean([set(true_predicates) == set(predicates) for true_predicates, predicates in zip(all_true_values, all_predicted_values)])

        accuracies = [
            dict(
                threshold=t,
                accuracy=get_accuracy(t),
                dataset=dataset,
                model=model,
                use_class_conditioned=use_class_conditioned,
                probe_name=probe_name
            )
            for t in range(5, 15)
        ]
        all_accuracies.extend(accuracies)
    return all_accuracies

def attach_d(rows, d):
    return [
        {**row, **d}
        for row in rows
    ]

def get_best_threshold(model, use_class_conditioned):
    all_accuracies = [
        attach_d(accuracy_sweep(dataset, model, use_class_conditioned), {'dataset': dataset})
        for dataset in ['basic', 'paraphrase', 'paraphrase_es']
    ]

    df = pd.DataFrame(sum(all_accuracies, []))

    def multi_index_map(df, func):
        def recurse(level, values):
            if level == len(df.index.levels):
                return func(*values)
            return [
                recurse(level + 1, values + (x,))
                for x in df.index.levels[level].values
            ]
        return recurse(0, ())


    def best_threshold(df, probe_name):
        probe_df = df[df['probe_name'] == probe_name]

        multi_df = probe_df.pivot(index=['dataset', 'threshold'], columns=[])

        accuracies = multi_index_map(multi_df, lambda dataset, threshold: multi_df.loc[(dataset, threshold), 'accuracy'])

        return np.array(accuracies).mean(axis=0).argmax() + multi_df.index.levels[1].min()

    thresholds = {
        probe_name: best_threshold(df, probe_name)
        for probe_name in ['name_probe', 'country_probe', 'occupation_probe', 'food_probe']
    }

    return thresholds, df

def set_threshold(model, use_class_conditioned, thresholds, accuracies):
    params = [
        ('name_probe', centroid_probe.NameCentroidProbe),
        ('country_probe', centroid_probe.CountryCentroidProbe),
        ('occupation_probe', centroid_probe.OccupationCentroidProbe),
        ('food_probe', centroid_probe.FoodCentroidProbe),
    ]
    probe_dir = Path(get_probe_save_dir('basic', model, use_class_conditioned))
    accuracies.to_csv(probe_dir/'accuracies.csv')
    
    for probe_name, probe_class in params:
        probe_path =  probe_dir / f'{probe_name}.pt'
        probe = probe_class.load_or_none(probe_path)
        probe.threshold = thresholds[probe_name]
        probe.save(probe_path)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expts_root', type=str, required=True)
    parser.add_argument('--outputs_root', type=str, required=True)
    args = parser.parse_args()
    expts_root = Path(args.expts_root)
    outputs_root = Path(args.outputs_root)
    for model in ['tulu', 'llama']:
        print(f'Getting thresholds for {model}')
        thresholds, accuracies = get_best_threshold(model, use_class_conditioned=True)
        print(thresholds)
        set_threshold(model=model, use_class_conditioned=True, thresholds=thresholds, accuracies=accuracies)