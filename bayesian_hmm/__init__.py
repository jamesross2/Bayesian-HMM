#!/usr/bin/env python3
"""
Non-parametric Bayesian hidden Markov models. Hidden Markov models are generative
time series models. This package uses a non-parametric Bayesian estimation process
that uses dynamic numbers of latent states to avoid having to specify this number in
advance.
"""

from .hdphmm import HDPHMM
from .chain import Chain
import warnings

warnings.warn(
    "bayesian_hmm is in beta testing, may change at any time, and has no guarantee of accuracy"
)
