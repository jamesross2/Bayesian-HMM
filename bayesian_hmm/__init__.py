#!/usr/bin/env python3
"""
Non-parametric Bayesian hidden Markov models. Hidden Markov models are generative
time series models. This package uses a non-parametric Bayesian estimation process
that uses dynamic numbers of latent states to avoid having to specify this number in
advance.
"""

from .hdphmm import HDPHMM
from .chain import Chain
from .hierarchical_dirichlet_process.symbol import Symbol, EmptySymbol
from bayesian_hmm import hierarchical_dirichlet_process
import warnings

warnings.warn(
    "bayesian_hmm is in beta testing and future versions may behave differently"
)
