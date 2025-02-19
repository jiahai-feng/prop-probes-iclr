import os
import numpy as np
import pandas as pd
from coref import COREF_ROOT
from coref.datasets.domains.common import SimpleDomain


class FoodDomain(SimpleDomain):
    '''
    using token width = 2!!
    '''
    type = 'FOODS'
    token_width = 2
    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        with open(os.path.join(COREF_ROOT, 'src/coref/datasets/raw/foods.txt')) as f:
            foods = f.read()
        foods = sorted(list(set([x.split('.')[1].strip().lower() for x in foods.split('\n')])))
        filtered_foods = self.apply_filters(foods, [self.generate_filter(foods, token_width=self.token_width)])
        self.data = self.filter_split(filtered_foods, split)



class NounPhraseDomain(SimpleDomain):
    '''
    no fixed token width!!
    '''
    type = 'NOUN_PHRASES'
    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        with open(os.path.join(src/COREF_ROOT, 'coref/datasets/raw/noun_phrases.csv')) as f:
            nps = f.read()
        nps = sorted(list(set([x.strip() for x in nps.split('\n')])))
        self.data = self.filter_split(nps, split)

class VerbPhraseDomain(SimpleDomain):
    '''
    no fixed token width!!
    '''
    type = 'VERB_PHRASES'
    token_width = 2
    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        with open(os.path.join(COREF_ROOT, 'src/coref/datasets/raw/verb_phrases.csv')) as f:
            nps = f.read()
        nps = sorted(list(set([x.strip() for x in nps.split('\n')])))
        self.data = self.filter_split(nps, split)


class OccupationDomain(SimpleDomain):
    type = 'OCCUPATIONS'
    token_width = 1
    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        with open(os.path.join(COREF_ROOT, 'src/coref/datasets/raw/winobias/occupations.txt')) as f:
            occupations = f.read()
        occupations = occupations.strip().split('\n')
        filtered_occupations = self.apply_filters(occupations, [self.generate_filter(occupations, token_width=self.token_width)])
        self.data = self.filter_split(filtered_occupations, split)
        
class FemaleOccupationDomain(SimpleDomain):
    type = 'FEMALE_OCCUPATIONS'
    token_width = 1
    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        with open(os.path.join(COREF_ROOT, 'src/coref/datasets/raw/winobias/female_occupations.txt')) as f:
            occupations = f.read()
        occupations = occupations.strip().split('\n')
        filtered_occupations = self.apply_filters(occupations, [self.generate_filter(occupations, token_width=self.token_width)])
        self.data = self.filter_split(filtered_occupations, split)
        
class MaleOccupationDomain(SimpleDomain):
    type = 'MALE_OCCUPATIONS'
    token_width = 1
    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        with open(os.path.join(COREF_ROOT, 'src/coref/datasets/raw/winobias/male_occupations.txt')) as f:
            occupations = f.read()
        occupations = occupations.strip().split('\n')
        filtered_occupations = self.apply_filters(occupations, [self.generate_filter(occupations, token_width=self.token_width)])
        self.data = self.filter_split(filtered_occupations, split)
        
class NameDomain(SimpleDomain):
    type = "NAMES"

    def __init__(self, tokenizer_type, split=None):
        """
        split: None | "train" | "test"
        """
        super().__init__(tokenizer_type)
        df = pd.read_csv(
            os.path.join(COREF_ROOT, "src/coref/datasets/raw/new-top-firstNames.csv")
        )
        names = df.name.to_list()
        filtered_names = self.apply_filters(names, [self.generate_filter(names)])
        self.data = self.filter_split(filtered_names, split)

def get_gendered_names():
    '''
    
    Returns
        list of male names, list of female names
    '''
    raw_path = os.path.join(COREF_ROOT, "src/coref/datasets/raw/baby-names.csv")
    male_path = os.path.join(COREF_ROOT, "src/coref/datasets/raw/male-baby-names.csv")
    female_path = os.path.join(COREF_ROOT, "src/coref/datasets/raw/female-baby-names.csv")

    if os.path.isfile(male_path) and os.path.isfile(female_path):
        male_df = pd.read_csv(male_path)
        female_df = pd.read_csv(female_path)
        return male_df.name.to_list(), female_df.name.to_list()
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"baby-names not found at {raw_path}")
        
    df = pd.read_csv(
        os.path.join(COREF_ROOT, "src/coref/datasets/raw/baby-names.csv")
    )

    names = {}
    for row in df.itertuples():
        if row.name not in names:
            names[row.name] = {'boy': 0, 'girl': 0}
            
        names[row.name][row.sex] += row.percent

    male_names = []
    female_names = []
    for name, stats in names.items():
        if stats['boy'] > stats['girl'] * 10:
            male_names.append({
                'name': name,
                'boy': stats['boy'],
                'girl': stats['girl']
            })
        elif stats['girl'] > stats['boy'] * 10:
            female_names.append({
                'name': name,
                'boy': stats['boy'],
                'girl': stats['girl']
            })
    

    male_names.sort(key=lambda x: -x['boy'])
    female_names.sort(key=lambda x: -x['girl'])

    pd.DataFrame(male_names[:200]).to_csv(male_path, index=None)
    pd.DataFrame(female_names[:200]).to_csv(female_path, index=None)

    male_df = pd.read_csv(male_path)
    female_df = pd.read_csv(female_path)
    return male_df.name.to_list(), female_df.name.to_list()
class MaleNameDomain(SimpleDomain):
    type = "MALE_NAMES"

    def __init__(self, tokenizer_type, split=None):
        """
        split: None | "train" | "test"
        """
        super().__init__(tokenizer_type)
        male_names, _ = get_gendered_names()
        filtered_names = self.apply_filters(male_names, [self.generate_filter(male_names)])
        self.data = self.filter_split(filtered_names, split)
class FemaleNameDomain(SimpleDomain):
    type = "FEMALE_NAMES"

    def __init__(self, tokenizer_type, split=None):
        """
        split: None | "train" | "test"
        """
        super().__init__(tokenizer_type)
        male_names, female_names = get_gendered_names()
        filtered_names = self.apply_filters(female_names, [self.generate_filter(female_names)])
        self.data = self.filter_split(filtered_names, split)

class CapitalDomain(SimpleDomain):
    type = "CAPITALS"

    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)
        df = pd.read_csv(
            os.path.join(COREF_ROOT, "src/coref/datasets/raw/country-list.csv")
        )
        country_capital_pairs = [(r.country, r.capital) for r in df.itertuples()]
        filtered_pairs = self.apply_filters(
            country_capital_pairs,
            [
                self.generate_filter([x for x, _ in country_capital_pairs]),
                self.generate_filter([y for _, y in country_capital_pairs]),
            ],
        )
        self.data = self.filter_split(filtered_pairs, split)


class ColorDomain(SimpleDomain):
    type = "COLORS"

    def __init__(self, tokenizer_type, split=None):
        """
        split: None | "train" | "test"
        """
        super().__init__(tokenizer_type)
        df = pd.read_csv(os.path.join(COREF_ROOT, "src/coref/datasets/raw/colors.csv"))
        names = df.color.to_list()
        filtered_names = self.apply_filters(names, [self.generate_filter(names)])
        self.data = self.filter_split(filtered_names, split)


class HobbyDomain(SimpleDomain):
    type = "HOBBIES"

    def __init__(self, tokenizer_type, split=None):
        """
        split: None | "train" | "test"
        """
        super().__init__(tokenizer_type)
        df = pd.read_csv("src/coref/datasets/raw/hobbies.csv", header=None, delimiter="\t")
        names = [name.lower() for name in df[0].to_list()]
        filtered_names = self.apply_filters(names, [self.generate_filter(names)])
        self.data = self.filter_split(filtered_names, split)


class AlphabetDomain(SimpleDomain):
    type = "ALPHABET"

    def __init__(self, tokenizer_type, split=None):
        """
        split: None | "train" | "test"
        """
        super().__init__(tokenizer_type)
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert len(alphabet) == 26
        self.data = self.filter_split(alphabet, split)


class ObjectDomain(SimpleDomain):
    type = "OBJECTS"

    def __init__(self, tokenizer_type, split=None):
        """
        split: None | "train" | "test"
        """
        super().__init__(tokenizer_type)
        df = pd.read_csv(
            "src/coref/datasets/raw/objects_with_bnc_frequency.csv", header=0, delimiter=","
        )
        objects = [obj.lower() for obj in df["object_name"].to_list()]
        self.data = self.apply_filters(objects, [self.generate_filter(objects)])
