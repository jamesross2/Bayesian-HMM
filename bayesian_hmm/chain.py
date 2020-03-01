#!/usr/bin/env python3
"""
The Chain object stores a single emission series.
It has methods to initialise the object, resample the latent states, and some convenient
printing methods.
"""

import copy
import random
import typing

import numpy

from . import bayesian_model

# Shorthand for numeric types.
Numeric = typing.Union[int, float]

# Oft-used dictionary initializations with shorthands.
DictStrNum = typing.Dict[bayesian_model.State, Numeric]
InitDict = DictStrNum
DictStrDictStrNum = typing.Dict[bayesian_model.State, DictStrNum]
NestedInitDict = DictStrDictStrNum


# Chain stores a single markov emission sequence plus associated latent variables
class Chain(object):
    """Store observed emission sequence and current latent sequence for a HMM."""

    def __init__(self, sequence: typing.Sequence[bayesian_model.State]) -> None:
        """Create a Hidden Markov Chain for an observed emission sequence.

        Args:
            sequence: An iterable containing observed emissions.
        """
        # initialise & store sequences
        self.emission_sequence: typing.List[bayesian_model.State] = copy.deepcopy(list(sequence))
        self.latent_sequence: typing.List[bayesian_model.State] = [bayesian_model.AggregateState() for _ in sequence]

        # calculate dependent hyperparameters
        self.T = len(self.emission_sequence)

        # keep flag to track initialisation
        self._initialised_flag = False

    def __len__(self) -> int:
        return self.T

    @property
    def initialised_flag(self) -> bool:
        """Test whether a Chain is initialised.

        Returns:
            True if the chain is initialised.
        """
        return self._initialised_flag

    @initialised_flag.setter
    def initialised_flag(self, value: bool) -> None:
        if value is True:
            raise RuntimeError("Chain must be initialised through initialise_chain method")
        elif value is False:
            self._initialised_flag = False
        else:
            raise ValueError("Initialised flag must be Boolean")

    def __repr__(self) -> str:
        return "<Chain, size {0}>".format(self.T)

    def __str__(self, print_len: int = 15) -> str:
        print_len = min(print_len - 1, self.T - 1)
        return "bayesian_hmm.Chain, size={T}, seq={s}".format(
            T=self.T,
            s=["{s}:{e}".format(s=s, e=e) for s, e in zip(self.latent_sequence, self.emission_sequence)][:print_len]
            + ["..."],
        )

    def tabulate(self) -> numpy.array:
        """Convert the latent and emission sequences into a single numpy array.

        Returns:
            A numpy array with shape (T, 2), where T is the length of the Chain.
        """
        return numpy.column_stack(
            [[symbol.value for symbol in sequence] for sequence in (self.latent_sequence, self.emission_sequence)]
        )

    # introduce randomly sampled states for all latent variables in Chain
    def initialise(self, states: typing.Set[bayesian_model.State]) -> None:
        """Initialise the chain by sampling latent states.

        Args:
            states: The states to draw from for latent variables.

        """
        # create latent sequence
        self.latent_sequence = random.choices(list(states), k=self.T)

        # update observations
        self._initialised_flag = True

    def log_likelihood(self, emission_probabilities: NestedInitDict, transition_probabilities: NestedInitDict) -> float:
        """Negative log likelihood of the Chain, using the given parameters.

        Args:
            emission_probabilities: The current probability that state s emits emission e.
            transition_probabilities: The current probability that state s0 is followed by state s1.

        Returns:
            A float for the log likelihood of the chain.

        """
        # edge case: zero-length sequence
        if self.T == 0:
            return 0.0

        # get probability of transition & of emission, at start and remaining time steps
        # np.prod([])==1, so this is safe
        starting_likelihood = transition_probabilities[bayesian_model.StartingState()][self.latent_sequence[0]]
        transition_likelihoods = [
            transition_probabilities[self.latent_sequence[t]][self.latent_sequence[t + 1]] for t in range(self.T - 1)
        ]
        emission_likelihoods = [
            emission_probabilities[self.latent_sequence[t]][self.emission_sequence[t]] for t in range(self.T)
        ]
        log_likelihoods = (
            numpy.log(starting_likelihood),
            sum(numpy.log(transition_likelihoods)),
            sum(numpy.log(emission_likelihoods)),
        )
        return sum(log_likelihoods)


def resample_latent_sequence(
    sequences: typing.Tuple[typing.List[bayesian_model.State], typing.List[bayesian_model.State]],
    states: typing.Set[bayesian_model.State],
    emission_probabilities: NestedInitDict,
    transition_probabilities: NestedInitDict,
) -> typing.List[bayesian_model.State]:
    """Resample the latent sequence of a Chain.

    This is usually called by another method or class, rather than directly. It
    is included to allow for multithreading in the resampling step.

    Args:
        sequences: An emission sequence and latent beam variables.
        states: States to choose from.
        emission_probabilities: Probabilities as given by the parent model.
        transition_probabilities: Probabilities as given by the parent model.

    Returns:
        A sequence of resampled latent variables. These have the same length as the input
            sequence, but are drawn with the hierarchical Dirichlet process distribution
            for both the emissions and transitions applied.
    """
    # extract size information
    emission_sequence, latent_sequence = sequences
    seqlen = len(emission_sequence)

    # edge case: zero-length sequence
    if seqlen == 0:
        return []

    # sample auxiliary beam variables
    starting_likelihood = transition_probabilities[bayesian_model.StartingState()][latent_sequence[0]]
    transition_likelihood = [starting_likelihood] + [
        transition_probabilities[latent_sequence[t]][latent_sequence[t + 1]] for t in range(seqlen - 1)
    ]
    auxiliary_vars = [numpy.random.uniform(0, p) for p in transition_likelihood]

    # initialise historical P(s_t | u_{1:t}, y_{1:t}) and latent sequence
    p_history: typing.List[typing.Dict[bayesian_model.State, Numeric]] = [dict()] * seqlen

    # compute probability of state t (currently the starting state t==0)
    p_history[0] = {
        s: (
            transition_probabilities[bayesian_model.StartingState()][s]
            * emission_probabilities[s][emission_sequence[0]]
            if starting_likelihood > auxiliary_vars[0]
            else 0
        )
        for s in states
    }

    # for remaining states, probabilities are function of emission and transition
    for t in range(1, seqlen):
        p_temp = {
            s2: (
                sum(p_history[t - 1][s1] for s1 in states if transition_probabilities[s1][s2] > auxiliary_vars[t])
                * emission_probabilities[s2][emission_sequence[t]]
            )
            for s2 in states
        }
        p_temp_total = sum(p_temp.values())
        p_history[t] = {s: p_temp[s] / p_temp_total for s in states}

    # overwrite latent sequence from end backwards
    latent_sequence[seqlen - 1] = random.choices(
        tuple(p_history[seqlen - 1].keys()), weights=tuple(p_history[seqlen - 1].values()), k=1
    )[0]

    # work backwards to compute new latent sequence
    for t in range(seqlen - 2, -1, -1):
        p_temp = {
            s1: p_history[t][s1] * transition_probabilities[s1][latent_sequence[t + 1]]
            if transition_probabilities[s1][latent_sequence[t + 1]] > auxiliary_vars[t + 1]
            else 0
            for s1 in states
        }
        latent_sequence[t] = random.choices(tuple(p_temp.keys()), weights=tuple(p_temp.values()), k=1)[0]

    # latent sequence now completely filled
    return latent_sequence
