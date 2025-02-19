import itertools
import torch
from collections import namedtuple
import pandas as pd
import os
import numpy as np
from coref.models import get_llama_tokenizer, get_pythia_tokenizer


def filter_fraction(n, frac):
    """
    n: int
    frac: int

    Returns np boolean mask
    """
    rng = np.random.default_rng(0)
    mid = int(n * frac) + 1
    return rng.permutation(n) < mid


class SimpleDomain:
    token_width = 1
    def __init__(self, tokenizer_type):
        """
        tokenizer_type: "pythia" or "llama"
        """
        if tokenizer_type == "pythia":
            self.LLAMA = False
        elif tokenizer_type == "llama":
            self.LLAMA = True
        else:
            raise Exception(f"Unknown tokenizer type {tokenizer_type}")
        if self.LLAMA:
            self.tokenizer = get_llama_tokenizer()
        else:
            self.tokenizer = get_pythia_tokenizer()
        self.conditionals = {}

    def apply_filters(self, words, filters):
        return [word for word, *flags in zip(words, *filters) if all(flags)]

    def generate_filter(self, words, token_width=1):
        return [self.get_length_of_word(word) == token_width for word in words]

    def encode_single_word(self, word, suppress_error=False):
        if self.LLAMA:
            stuff = self.tokenizer.encode(word)[1:]
        else:
            stuff = self.tokenizer.encode(" " + word)
        if not suppress_error:
            assert len(stuff) == 1
        return stuff[0]

    def get_length_of_word(self, word):
        if self.LLAMA:
            return len(self.tokenizer.encode(word)) - 1
        else:
            return len(self.tokenizer(" " + word)["input_ids"])

    @staticmethod
    def train_test_filter(num_words, split):
        train_ratio = 0.5
        if split == "train":
            return filter_fraction(num_words, train_ratio).tolist()
        elif split == "test":
            return (~filter_fraction(num_words, train_ratio)).tolist()
        else:
            assert split is None
            return [True] * num_words

    def filter_split(self, tuples, split):
        return self.apply_filters(tuples, [self.train_test_filter(len(tuples), split)])

    def set_perm(self, prompt_id):
        mod = 1_000_000_007

        def bad_hash(s):
            r = 0
            for i in s:
                r *= 34389
                r += ord(i)
                r %= mod
            return r

        rng = np.random.default_rng(prompt_id + bad_hash(self.type) % mod)
        c_perm = rng.permutation(len(self.data) - len(self.conditionals))
        conditional_labels = set(self.conditionals.keys())
        conditional_values = set(self.conditionals.values())
        label_map = []
        value_map = []
        for i in range(len(self.data)):
            if not i in conditional_labels:
                label_map.append(i)
            if not i in conditional_values:
                value_map.append(i)
        self.perm = np.arange(len(self.data))
        for old_label, old_value in enumerate(c_perm):
            self.perm[label_map[old_label]] = value_map[old_value]
        for label, value in self.conditionals.items():
            self.perm[label] = value

    def lookup(self, i, perm=None):
        if perm is None:
            perm = self.perm
        return self.data[perm[i]]

    def __getitem__(self, i):
        return self.data[self.perm[i]]

    def __len__(self):
        return len(self.data)


from contextlib import contextmanager


@contextmanager
def set_prompt_id(prompt_id, *domains):
    for d in domains:
        d.set_perm(prompt_id)
    try:
        yield
    finally:
        for d in domains:
            d.perm = None

@contextmanager
def condition_domain(domain, label, underlying):
    '''
    We condition label to always be underlying
    '''
    assert label not in domain.conditionals.keys()
    assert underlying not in domain.conditionals.values()
    domain.conditionals[label] = underlying
    try:
        yield
    finally:
        domain.conditionals.pop(label)
