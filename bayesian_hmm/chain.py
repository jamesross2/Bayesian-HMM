#!/usr/bin/env python3
"""
The Chain object stores a single emission series.
It has methods to initialise the object, resample the latent states, and some convenient
printing methods.
"""
# Support typehinting.
from __future__ import annotations
from typing import (
    Collection,
    Sequence,
    Dict,
    Any,
    Union,
    List,
    Tuple,
    Set,
    Optional,
    Sized,
    Iterable,
)

import numpy as np
import random
import copy

# Shorthand for numeric types.
Numeric = Union[int, float]

# Oft-used dictionary initializations with shorthands.
DictStrNum = Dict[Optional[str], Numeric]
InitDict = DictStrNum
DictStrDictStrNum = Dict[Optional[str], DictStrNum]
NestedInitDict = DictStrDictStrNum


# Chain stores a single markov emission sequence plus associated latent variables
class Chain(object):
    """
    Store observed emission sequence and current latent sequence for a HMM.
    """

    def __init__(self, sequence: List[Optional[str]]) -> None:
        """
        Create a Hidden Markov Chain for an observed emission sequence.
        :param sequence: iterable containing observed emissions.
        """
        # initialise & store sequences
        self.emission_sequence: List[Optional[str]] = copy.deepcopy(sequence)
        self.latent_sequence: List[Optional[str]] = [None for _ in sequence]

        # calculate dependent hyperparameters
        self.T = len(self.emission_sequence)

        # keep flag to track initialisation
        self._initialised_flag = False

    def __len__(self) -> int:
        return self.T

    @property
    def initialised_flag(self) -> bool:
        """
        Test whether a Chain is initialised.
        :return: bool
        """
        return self._initialised_flag

    @initialised_flag.setter
    def initialised_flag(self, value: bool) -> None:
        if value is True:
            raise RuntimeError(
                "Chain must be initialised through initialise_chain method"
            )
        elif value is False:
            self._initialised_flag = False
        else:
            raise ValueError("initialised flag must be Boolean")

    def __repr__(self) -> str:
        return "<bayesian_hmm.Chain, size {0}>".format(self.T)

    def __str__(self, print_len: int = 15) -> str:
        print_len = min(print_len - 1, self.T - 1)
        return "bayesian_hmm.Chain, size={T}, seq={s}".format(
            T=self.T,
            s=[
                "{s}:{e}".format(s=s, e=e)
                for s, e in zip(self.latent_sequence, self.emission_sequence)
            ][:print_len]
            + ["..."],
        )

    def tabulate(self) -> np.array:
        """
        Convert the latent and emission sequences into a single numpy array.
        :return: numpy array with shape (l, 2), where l is the length of the Chain
        """
        return np.column_stack(
            (copy.copy(self.latent_sequence), copy.copy(self.emission_sequence))
        )

    # introduce randomly sampled states for all latent variables in Chain
    def initialise(self, states: Sequence) -> None:
        """
        Initialise the chain by sampling latent states.
        Typically called directly from an HDPHMM object.
        :param states: set of states to sample from
        :return: None
        """
        # create latent sequence
        self.latent_sequence = random.choices(states, k=self.T)

        # update observations
        self._initialised_flag = True

    def neglogp_chain(
        self,
        p_initial: InitDict,
        p_emission: NestedInitDict,
        p_transition: NestedInitDict,
    ) -> Numeric:
        """
        Negative log likelihood of the Chain, using the given parameters.
        Usually called with parameters given by the parent HDPHMM object.
        :param p_initial: dict, initial probabilities
        :param p_emission: dict, emission probabilities
        :param p_transition: dict, transition probabilities
        :return: float
        """
        # edge case: zero-length sequence
        if self.T == 0:
            return 0

        # get probability of transition & of emission, at start and remaining time steps
        # np.prod([])==1, so this is safe
        p_start = np.log(p_initial[self.latent_sequence[0]]) + np.log(
            p_emission[self.latent_sequence[0]][self.emission_sequence[0]]
        )
        p_remainder = [
            np.log(p_emission[self.latent_sequence[t]][self.emission_sequence[t]])
            + np.log(p_transition[self.latent_sequence[t - 1]][self.latent_sequence[t]])
            for t in range(1, self.T)
        ]

        # take log and sum for result
        return -(p_start + sum(p_remainder))

    @staticmethod
    def resample_latent_sequence(
        sequences: Tuple[List[str], List[str]],
        states: Set[str],
        p_initial: InitDict,
        p_emission: NestedInitDict,
        p_transition: NestedInitDict,
    ) -> List[str]:
        """
        Resample the latent sequence of a Chain. This is usually called by another
        method or class, rather than directly. It is included to allow for
        multithreading in the resampling step.
        :param sequences: tuple(list, list), an emission sequence and latent beam
        variables
        :param states: set, states to choose from
        :param p_initial: dict, initial probabilities
        :param p_emission: dict, emission probabilities
        :param p_transition: dict, transition probabilities
        :return: list, resampled latent sequence
        """
        # extract size information
        emission_sequence, latent_sequence = sequences
        seqlen = len(emission_sequence)

        # edge case: zero-length sequence
        if seqlen == 0:
            return []

        # sample auxiliary beam variables
        temp_p_transition = [p_initial[latent_sequence[0]]] + [
            p_transition[latent_sequence[t]][latent_sequence[t + 1]]
            for t in range(seqlen - 1)
        ]
        auxiliary_vars = [np.random.uniform(0, p) for p in temp_p_transition]

        # initialise historical P(s_t | u_{1:t}, y_{1:t}) and latent sequence
        p_history: List[Dict[str, Numeric]]
        p_history = [dict()] * seqlen
        latent_sequence = [str()] * seqlen

        # compute probability of state t (currently the starting state t==0)
        p_history[0] = {
            s: p_initial[s] * p_emission[s][emission_sequence[0]]
            if p_initial[s] > auxiliary_vars[0]
            else 0
            for s in states
        }

        # for remaining states, probabilities are function of emission and transition
        for t in range(1, seqlen):
            p_temp = {
                s2: sum(
                    p_history[t - 1][s1]
                    for s1 in states
                    if p_transition[s1][s2] > auxiliary_vars[t]
                )
                * p_emission[s2][emission_sequence[t]]
                for s2 in states
            }
            p_temp_total = sum(p_temp.values())
            p_history[t] = {s: p_temp[s] / p_temp_total for s in states}

        # choose ending state
        latent_sequence[seqlen - 1] = random.choices(
            tuple(p_history[seqlen - 1].keys()),
            weights=tuple(p_history[seqlen - 1].values()),
            k=1,
        )[0]

        # work backwards to compute new latent sequence
        for t in range(seqlen - 2, -1, -1):
            p_temp = {
                s1: p_history[t][s1] * p_transition[s1][latent_sequence[t + 1]]
                if p_transition[s1][latent_sequence[t + 1]] > auxiliary_vars[t + 1]
                else 0
                for s1 in states
            }
            latent_sequence[t] = random.choices(
                tuple(p_temp.keys()), weights=tuple(p_temp.values()), k=1
            )[0]

        # latent sequence now completely filled
        return latent_sequence
