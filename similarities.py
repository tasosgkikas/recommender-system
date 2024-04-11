from math import sqrt
from typing import Any, Dict


FloatValueDict = Dict[Any, float]


def jaccard(x: FloatValueDict, y: FloatValueDict) -> float:
    set_x = set(x)
    set_y = set(y)
    intersection = set_x.intersection(set_y)
    union = set_x.union(set_y)
    denominator = len(union)
    if denominator == 0: return 0
    return len(intersection) / denominator


def dice(x: FloatValueDict, y: FloatValueDict) -> float:
    set_x = set(x)
    set_y = set(y)
    intersection = set_x.intersection(set_y)
    denominator = len(set_x) + len(set_y)
    if denominator == 0: return 0
    return 2*len(intersection) / denominator


def _dict_norm(x: FloatValueDict) -> float:
    return sqrt(sum([val*val for val in x.values()]))

def _dict_dot_product(x: FloatValueDict, y: FloatValueDict) -> float:
    keys = set(x).intersection(set(y))
    return sum([x[key] * y[key] for key in keys])

def cosine(x: FloatValueDict, y: FloatValueDict) -> float:
    denominator = round(_dict_norm(x) * _dict_norm(y), 10)
    if denominator == 0: return 0
    return _dict_dot_product(x, y) / denominator


def _zero_centered(x: FloatValueDict) -> FloatValueDict:
    mean = sum(x.values()) / len(x)
    return {key: (x[key] - mean) for key in x}

def pearson(x: FloatValueDict, y: FloatValueDict) -> float:
    return cosine(*map(_zero_centered, (x, y)))
