#!/usr/bin/env python3
"""
Non-parametric Bayesian hidden Markov models. Hidden Markov models are generative
time series models. This package uses a non-parametric Bayesian estimation process
that uses dynamic numbers of latent states to avoid having to specify this number in
advance.
"""

import warnings

from .bayesian_model import (
    AggregateState,
    DirichletFamily,
    DirichletProcessFamily,
    HierarchicalDirichletProcess,
    StartingState,
    State,
    StickBreakingProcess,
    Variable,
    hyperparameter,
)
from .chain import Chain
from .hdphmm import HDPHMM

warnings.warn("bayesian_hmm is in beta testing and future versions may behave differently")
