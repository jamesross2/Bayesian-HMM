#!/usr/bin/env python3
"""
Helper functions for the Bayesian-HMM package. Should not be called directly by the
user.
"""
# Support typehinting.
from __future__ import annotations

import itertools
import random
import string
import typing

import numpy as np

import bayesian_hmm


# used to give human-friendly labels to states as they are created
def label_generator(labels: str = string.ascii_lowercase) -> typing.Generator[str, None, None]:
    """Generate a non-repeating and intuitively incrementing series of labels.

    Args:
        labels: These will be used (consecutively) as prefixes on the labels. After
            the list has been iterated, numbers are appended. For this reason,
            labels should not be numeric.

    Yields:
        A string for the next label of the form
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
def dirichlet_process_generator(
    alpha: typing.Union[int, float] = 1, output_generator: typing.Iterator[typing.Union[str, int]] = None
) -> typing.Generator[typing.Union[str, int], None, None]:
    """Creates a generator object which yields subsequent draws from a single dirichlet process.

    Args:
        alpha: Standard parameter of the Dirichlet process.
        output_generator: A generator which yields unique symbols.

    Yields:
        Draws from a growing Dirichlet process.

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
            val = np.random.choice(list(weights.keys()), 1, p=list(x / count for x in weights.values()))[0]
            weights[val] += 1
        count += 1
        yield val


# used to ensure all hyperparameters have non-zero values
def max_dict(
    d: typing.Dict[str, typing.Union[int, float]], eps: float = 1e-8
) -> typing.Dict[bayesian_hmm.State, float]:
    return {k: max(float(v), eps) for k, v in d.items()}


def shrink_probabilities(d: typing.Iterable[float], eps: float = 1e-12) -> typing.Iterable[float]:
    if isinstance(d, dict):
        denom = sum(d.values()) + len(d) * eps
        return {k: (float(v) + eps) / denom for k, v in d.items()}
    elif isinstance(d, tuple):
        denom = sum(d) + eps * len(d)
        return tuple((float(v) + eps) / denom for v in d)
