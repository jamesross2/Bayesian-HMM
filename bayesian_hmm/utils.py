#!/usr/bin/env python3
"""
Helper functions for the Bayesian-HMM package. Should not be called directly by the
user.
"""

import numpy as np
import random
import itertools
import string


# used to give human-friendly labels to states as they are created
def label_generator(labels=string.ascii_lowercase):
    """
    :param labels: set of labels to choose from. Should not be numeric.
    :return: a generator which  yields unique labels of the form
    a, b, ..., z, a1, b1, ...
    """
    x, y, z = 0, 0, ""
    while True:
        if x < len(labels):
            yield labels[x] + z
            x += 1
        if x == len(labels):
            y += 1
            z = str(y)
            x = 0


# used to choose from new states after resampling latent states
def dirichlet_process_generator(alpha=1, output_generator=None):
    """
    Returns a generator object which yields subsequent draws from a single dirichlet
    process.
    :param alpha: alpha parameter of the Dirichlet process
    :param output_generator: generator which yields unique symbols
    :return: generator object
    """
    if output_generator is None:
        output_generator = itertools.count(start=0, step=1)
    count = 0
    weights = {}
    while True:
        if random.uniform(0, 1) > (count / (count + alpha)):
            val = next(output_generator)
            weights[val] = 1
        else:
            val = np.random.choice(
                list(weights.keys()), 1, p=list(x / count for x in weights.values())
            )[0]
            weights[val] += 1
        count += 1
        yield val


# used to ensure all hyperparameters have non-zero values
def max_dict(d, eps=1e-8):
    return {k: max(float(v), eps) for k, v in d.items()}


def shrink_probabilities(d, eps=1e-12):
    denom = sum(d.values()) + len(d) * eps
    return {k: (float(v) + eps) / denom for k, v in d.items()}
