#!/usr/bin/env python3
"""
Hierarchical Dirichlet Process Hidden Markov Model (HDPHMM).
The HDPHMM object collects a number of observed emission sequences, and estimates
latent states at every time point, along with a probability structure that ties latent
states to emissions. This structure involves

  + A starting probability, which dictates the probability that the first state
  in a latent seqeuence is equal to a given symbol. This has a hierarchical Dirichlet
  prior.
  + A transition probability, which dictates the probability that any given symbol in
  the latent sequence is followed by another given symbol. This shares the same
  hierarchical Dirichlet prior as the starting probabilities.
  + An emission probability, which dictates the probability that any given emission
  is observed conditional on the latent state at the same time point. This uses a
  Dirichlet prior.

Fitting HDPHMMs requires MCMC estimation. MCMC estimation is thus used to calculate the
posterior distribution for the above probabilities. In addition, we can use MAP
estimation (for example) to fix latent states, and facilitate further analysis of a
Chain.
"""
# Support typehinting.
from __future__ import annotations
from typing import Any, Union, Optional, Set, Dict, Iterable, List, Callable, Generator

import numpy as np
import random
import copy
import terminaltables
import tqdm
import functools
import multiprocessing
import string
from scipy import special, stats
from sympy.functions.combinatorial.numbers import stirling
from .chain import Chain
from .utils import label_generator, dirichlet_process_generator, shrink_probabilities
from warnings import catch_warnings

# Shorthand for numeric types.
Numeric = Union[int, float]

# Oft-used dictionary initializations with shorthands.
DictStrNum = Dict[Optional[str], Numeric]
InitDict = DictStrNum
DictStrDictStrNum = Dict[Optional[str], DictStrNum]
NestedInitDict = DictStrDictStrNum


class HDPHMM(object):
    """
    The Hierarchical Dirichlet Process Hidden Markov Model object. In fact, this is a
    sticky-HDPHMM, since we allow a biased self-transition probability.
    """

    def __init__(
        self,
        emission_sequences: Iterable[List[Optional[str]]],
        emissions=None,  # type: ignore
        # emissions: Optional[Iterable[Union[str, int]]] = None # ???
        sticky: bool = True,
        priors: Dict[str, Callable[[], Any]] = None,
    ) -> None:
        """
        Create a Hierarchical Dirichlet Process Hidden Markov Model object, which can
        (optionally) be sticky. The emission sequences must be provided, although all
        other parameters are initialised with reasonable default values. It is
        recommended to specify the `sticky` parameter, depending on whether you believe
        the HMM to have a high probability of self-transition.
        
        :param emission_sequences: iterable, containing the observed emission sequences.
        emission sequences can be different lengths, or zero length.
        
        :param emissions: set, optional. If not all emissions are guaranteed to be
        observed in the data, this can be used to force non-zero emission probabilities
        for unobserved emissions.
        
        :param sticky: bool, flag to indicate whether the HDPHMM is sticky or not.
        Sticky HDPHMMs have an additional value (kappa) added to the probability of self
        transition. It is recommended to set this depending on the knowledge of the
        problem at hand.
        
        :param priors: dict, containing priors for the model hyperparameters. Priors
        should be functions with zero arguments. The following priors are accepted:
          + alpha: prior distribution of the alpha parameter. Alpha
            parameter is the value used in the hierarchical Dirichlet prior for
            transitions and starting probabilities. Higher values of alpha keep rows of
            the transition matrix more similar to the beta parameters.
          + gamma: prior distribution of the gamma parameter. Gamma controls the
            strength of the uninformative prior in the starting and transition
            distributions. Hence, it impacts the likelihood of resampling unseen states
            when estimating beta coefficients. That is, higher values of gamma mean the
            HMM is more likely to explore new states when resampling.
          + alpha_emission: prior distribution of the alpha parameter for the
            emission prior distribution. Alpha controls how tightly the conditional
            emission distributions follow their hierarchical prior. Hence, higher values
            of alpha_emission mean more strength in the hierarchical prior.
          + gamma_emission: prior distribution of the gamma parameter for the
            emission prior distribution. Gamma controls the strength of the
            uninformative prior in the emission distribution. Hence, higher values of
            gamma mean more strength of belief in the prior.
          + kappa: prior distribution of the kappa parameter for the
            self-transition probability. Ignored if `sticky==False`. Kappa prior should
            have support in (0, 1) only. Higher values of kappa mean the chain is more
            likely to explore states with high self-transition probabilty.
        """
        # store chains
        self.chains = [Chain(sequence) for sequence in emission_sequences]

        # sticky flag
        if type(sticky) is not bool:
            raise ValueError("`sticky` must be type bool")
        self.sticky = sticky

        # store hyperparameter priors
        self.priors = {
            "alpha": lambda: np.random.gamma(2, 2),
            "gamma": lambda: np.random.gamma(3, 3),
            "alpha_emission": lambda: np.random.gamma(2, 2),
            "gamma_emission": lambda: np.random.gamma(3, 3),
            "kappa": lambda: np.random.beta(1, 1),
        }
        if priors is not None:
            self.priors.update(priors)
        if len(self.priors) > 5:
            raise ValueError("Unknown hyperparameter priors present")

        if not self.sticky:
            self.priors["kappa"] = lambda: 0
            if priors is not None and "kappa" in priors:
                raise ValueError("`sticky` is False, but kappa prior function given")

        # store initial hyperparameter values
        self.hyperparameters = {param: prior() for param, prior in self.priors.items()}

        # use internal properties to store fit hyperparameters
        self.n_initial: InitDict
        self.n_emission: NestedInitDict
        self.n_transition: NestedInitDict
        self.n_initial = {None: 0}
        self.n_emission = {None: {None: 0}}
        self.n_transition = {None: {None: 0}}

        # use internal properties to store current state for probabilities
        self.p_initial: InitDict
        self.p_emission: NestedInitDict
        self.p_transition: NestedInitDict
        self.p_initial = {None: 1}
        self.p_emission = {None: {None: 1}}
        self.p_transition = {None: {None: 1}}

        # store derived hyperparameters
        self.auxiliary_transition_variables: NestedInitDict
        self.beta_transition: InitDict
        self.beta_emission: InitDict
        self.auxiliary_transition_variables = {None: {None: 0}}
        self.beta_transition = {None: 1}
        self.beta_emission = {None: 1}

        # states & emissions
        # TODO: figure out emissions's type...
        if emissions is None:
            emissions = functools.reduce(  # type: ignore
                set.union, (set(c.emission_sequence) for c in self.chains), set()
            )
        elif not isinstance(emissions, set):
            raise ValueError("emissions must be a set")
        self.emissions = emissions  # type: ignore
        self.states: Set[Optional[str]] = set()

        # generate non-repeating character labels for latent states
        self._label_generator = label_generator(string.ascii_lowercase)

        # keep flag to track initialisation
        self._initialised = False

    @property
    def initialised(self) -> bool:
        """
        Test whether a HDPHMM is initialised.
        :return: bool
        """
        return self._initialised

    @initialised.setter
    def initialised(self, value: Any) -> None:
        if value:
            raise AssertionError("HDPHMM must be initialised through initialise method")
        elif not value:
            self._initialised = False
        else:
            raise ValueError("initialised flag must be Boolean")

    @property
    def c(self) -> int:
        """
        Number of chains in the HMM.
        :return: int
        """
        return len(self.chains)

    @property
    def k(self) -> int:
        """
        Number of latent states in the HMM currently.
        :return: int
        """
        return len(self.states)

    @property
    def n(self) -> int:
        """
        Number of unique emissions. If `emissions` was specified when the HDPHMM was
        created, then this counts the number of elements in `emissions`. Otherwise,
        counts the number of observed emissions across all emission sequences.
        :return: int
        """
        return len(self.emissions)

    def tabulate(self) -> np.array:
        """
        Convert the latent and emission sequences for all chains into a single numpy
        array. Array contains an index which matches a Chain's index in
        HDPHMM.chains, the current latent state, and the emission for all chains at
        all times.
        :return: numpy array with dimension (l, 3), where l is the length of the Chain
        """
        hmm_array = np.concatenate(
            tuple(
                np.concatenate(
                    (np.array([[n] * self.chains[n].T]).T, self.chains[n].tabulate()),
                    axis=1,
                )
                for n in range(self.c)
            ),
            axis=0,
        )
        return hmm_array

    def __repr__(self) -> str:
        return "<bayesian_hmm.HDPHMM, size {C}>".format(C=self.c)

    def __str__(self, print_len: int = 15) -> str:
        fs = (
            "bayesian_hmm.HDPHMM,"
            + " ({C} chains, {K} states, {N} emissions, {Ob} observations)"
        )
        return fs.format(C=self.c, K=self.k, N=self.n, Ob=sum(c.T for c in self.chains))

    def state_generator(self, eps: Numeric = 1e-12) -> Generator[str, None, None]:
        """
        Create a new state for the HDPHMM, and update all parameters accordingly.
        This involves updating
          + The counts for the new symbol
          + The auxiliary variables for the new symbol
          + The probabilities for the new symbol
          + The states captured by the HDPHMM
        :return: str, label of the new state
        """
        while True:
            label = next(self._label_generator)

            # update counts with zeros (assume _n_update called later)
            # state irrelevant for constant count (all zeros)
            self.n_initial[label] = 0
            self.n_transition[label] = {s: 0 for s in self.states.union({label, None})}
            for s in self.states:
                self.n_transition[s].update({label: 0})
            self.n_emission[label] = {e: 0 for e in self.emissions}

            # update auxiliary transition variables
            self.auxiliary_transition_variables[label] = {
                s2: 1 for s2 in list(self.states) + [label, None]
            }
            for s1 in self.states:
                self.auxiliary_transition_variables[s1][label] = 1

            # update beta_transition value and split out from current pseudo state
            temp_beta = np.random.beta(1, self.hyperparameters["gamma"])
            self.beta_transition[label] = temp_beta * self.beta_transition[None]
            self.beta_transition[None] = (1 - temp_beta) * self.beta_transition[None]

            # update starting probability
            temp_p_initial = np.random.beta(1, self.hyperparameters["gamma"])
            self.p_initial[label] = temp_p_initial * self.p_initial[None]
            self.p_initial[None] = (1 - temp_p_initial) * self.p_initial[None]

            # update transition from new state
            temp_p_transition = np.random.dirichlet(
                [self.beta_transition[s] for s in list(self.states) + [label, None]]
            )
            p_transition_label = dict(
                zip(list(self.states) + [label, None], temp_p_transition)
            )
            self.p_transition[label] = shrink_probabilities(p_transition_label, eps)

            # update transitions into new state
            for state in self.states.union({None}):
                # (note that label not included in self.states)
                temp_p_transition = np.random.beta(1, self.hyperparameters["gamma"])
                self.p_transition[state][label] = (
                    self.p_transition[state][None] * temp_p_transition
                )
                self.p_transition[state][None] = self.p_transition[state][None] * (
                    1 - temp_p_transition
                )

            # update emission probabilities
            temp_p_emission = np.random.dirichlet(
                [
                    self.hyperparameters["alpha"] * self.beta_emission[e]
                    for e in self.emissions
                ]
            )
            self.p_emission[label] = dict(zip(self.emissions, temp_p_emission))

            # save label
            self.states = self.states.union({label})

            yield label

    def initialise(self, k: int = 20) -> None:
        """
        Initialise the HDPHMM. This involves:
          + Choosing starting values for all hyperparameters
          + Initialising all Chains (see Chain.initialise for further info)
          + Initialising priors for probabilities (i.e. the Hierarchical priors)
          + Updating all counts
        
        sampling latent states, auxiliary beam variables,
        Typically called directly from a HDPHMM object.
        :param k: number of symbols to sample from for latent states
        :return: None
        """
        # create as many states as needed
        states = [next(self._label_generator) for _ in range(k)]
        self.states = set(states)

        # set hyperparameters
        self.hyperparameters = {param: prior() for param, prior in self.priors.items()}

        # initialise chains
        for c in self.chains:
            c.initialise(states)

        # initialise hierarchical priors
        temp_beta = sorted(
            np.random.dirichlet(
                [self.hyperparameters["gamma"] / (self.k + 1)] * (self.k + 1)
            ),
            reverse=True,
        )
        beta_transition = dict(zip(list(self.states) + [None], temp_beta))
        self.beta_transition = shrink_probabilities(beta_transition)
        self.auxiliary_transition_variables = {
            s1: {s2: 1 for s2 in self.states.union({None})}
            for s1 in self.states.union({None})
        }

        # update counts before resampling
        self._n_update()

        # resample remaining hyperparameters
        self.resample_beta_transition()
        self.resample_beta_emission()
        self.resample_p_initial()
        self.resample_p_transition()
        self.resample_p_emission()

        # set initialised flag
        self._initialised = True

    def update_states(self):
        """
        Remove defunct states from the internal set of states, and merge all parameters
        associated with these states back into the 'None' values.
        """
        # identify states to merge
        states_prev = self.states
        states_next = set(
            sorted(
                functools.reduce(
                    set.union, (set(c.latent_sequence) for c in self.chains), set()
                )
            )
        )
        states_removed = states_prev - states_next

        # merge old probabilities into None
        for state in states_removed:
            # remove entries and add to aggregate None state
            self.beta_transition[None] += self.beta_transition.pop(state)
            self.p_initial[None] += self.p_initial.pop(state)
            for s1 in states_next.union({None}):
                self.p_transition[s1][None] += self.p_transition[s1].pop(state)

            # remove transition vector entirely
            del self.p_transition[state]

        # update internal state tracking
        self.states = states_next

    def _n_update(self):
        """
        Update counts required for resampling probabilities. These counts are used
        to sample from the posterior distribution for probabilities. This function
        should be called after any latent state is changed, including after resampling.
        :return: None
        """
        # check that all chains are initialised
        if any(not chain.initialised_flag for chain in self.chains):
            raise AssertionError(
                "Chains must be initialised before calculating fit parameters"
            )

        # transition count for non-oracle transitions
        n_initial = {s: 0 for s in self.states.union({None})}
        n_emission = {
            s: {e: 0 for e in self.emissions} for s in self.states.union({None})
        }
        n_transition = {
            s1: {s2: 0 for s2 in self.states.union({None})}
            for s1 in self.states.union({None})
        }

        # increment all relevant hyperparameters while looping over sequence
        for chain in self.chains:
            # increment initial states emitted by oracle
            n_initial[chain.latent_sequence[0]] += 1

            # increment emissions only for final state
            n_emission[chain.latent_sequence[chain.T - 1]][
                chain.emission_sequence[chain.T - 1]
            ] += 1

            # increment all transitions and emissions within chain
            for t in range(chain.T - 1):
                n_emission[chain.latent_sequence[t]][chain.emission_sequence[t]] += 1
                n_transition[chain.latent_sequence[t]][
                    chain.latent_sequence[t + 1]
                ] += 1

        # store recalculated fit hyperparameters
        self.n_initial = n_initial
        self.n_emission = n_emission
        self.n_transition = n_transition

    @staticmethod
    def _resample_auxiliary_transition_atom_complete(
        alpha, beta, n, use_approximation=True
    ):
        """
        Use a resampling approach that estimates probabilities for all auxiliary
        transition parameters. This avoids the slowdown in convergence caused by
        Metropolis Hastings rejections, but is more computationally costly.
        :param alpha:
        :param beta:
        :param n:
        :param use_approximation:
        :return:
        """
        # initialise values required to resample
        p_required = np.random.uniform(0, 1)
        m = 0
        p_cumulative = 0
        scale = alpha * beta

        if not use_approximation:
            # use precise probabilities
            try:
                logp_constant = np.log(special.gamma(scale)) - np.log(
                    special.gamma(scale + n)
                )
                while p_cumulative == 0 or p_cumulative < p_required and m < n:
                    # accumulate probability
                    m += 1
                    logp_accept = (
                        m * np.log(scale)
                        + np.log(stirling(n, m, kind=1))
                        + logp_constant
                    )
                    p_cumulative += np.exp(logp_accept)
            # after one failure use only the approximation
            except (RecursionError, OverflowError):
                # correct for failed case before
                m -= 1
        while p_cumulative < p_required and m < n:
            # problems with stirling recursion (large n & m), use approximation instead
            # magic number is the Euler constant
            # approximation derived in documentation
            m += 1
            logp_accept = (
                m
                + (m + scale - 0.5) * np.log(scale)
                + (m - 1) * np.log(0.57721 + np.log(n - 1))
                - (m - 0.5) * np.log(m)
                - scale * np.log(scale + n)
                - scale
            )
            p_cumulative += np.exp(logp_accept)
        # breaks out of loop after m is sufficiently large
        return max(m, 1)

    @staticmethod
    def _resample_auxiliary_transition_atom_mh(
        alpha, beta, n, m_curr, use_approximation=True
    ):
        """
        Use a Metropolos Hastings resampling approach that often rejects the proposed
        value. This can cause the convergence to slow down (as the values are less
        dynamic) but speeds up the computation.
        :param alpha:
        :param beta:
        :param n:
        :param m_curr:
        :param use_approximation:
        :return:
        """
        # propose new m
        n = max(n, 1)
        m_proposed = random.choice(range(1, n + 1))
        if m_curr > n:
            return m_proposed

        # find relative probabilities
        if use_approximation and n > 10:
            logp_diff = (
                (m_proposed - 0.5) * np.log(m_proposed)
                - (m_curr - 0.5) * np.log(m_curr)
                + (m_proposed - m_curr) * np.log(alpha * beta * np.exp(1))
                + (m_proposed - m_curr) * np.log(0.57721 + np.log(n - 1))
            )
        else:
            p_curr = float(stirling(n, m_curr, kind=1)) * ((alpha * beta) ** m_curr)
            p_proposed = float(stirling(n, m_proposed, kind=1)) * (
                (alpha * beta) ** m_proposed
            )
            logp_diff = np.log(p_proposed) - np.log(p_curr)

        # use MH variable to decide whether to accept m_proposed
        with catch_warnings(record=True) as caught_warnings:
            p_accept = min(1, np.exp(logp_diff))
            p_accept = bool(np.random.binomial(n=1, p=p_accept))  # convert to boolean
            if caught_warnings:
                p_accept = True

        return m_proposed if p_accept else m_curr

    @staticmethod
    def _resample_auxiliary_transition_atom(
        state_pair,
        alpha,
        beta,
        n_initial,
        n_transition,
        auxiliary_transition_variables,
        resample_type="mh",
        use_approximation=True,
    ):
        """
        Resampling the auxiliary transition atoms should be performed before resampling
        the transition beta values. This is the static method, creating to allow for
        parallelised resampling.
        :param state_pair:
        :param alpha:
        :param beta:
        :param n_initial:
        :param n_transition:
        :param auxiliary_transition_variables:
        :param resample_type:
        :param use_approximation:
        :return:
        """
        # extract states
        state1, state2 = state_pair

        # apply resampling
        if resample_type == "mh":
            return HDPHMM._resample_auxiliary_transition_atom_mh(
                alpha,
                beta[state2],
                n_initial[state2] + n_transition[state1][state2],
                auxiliary_transition_variables[state1][state2],
                use_approximation,
            )
        elif resample_type == "complete":
            return HDPHMM._resample_auxiliary_transition_atom_complete(
                alpha,
                beta[state2],
                n_initial[state2] + n_transition[state1][state2],
                use_approximation,
            )
        else:
            raise ValueError("resample_type must be either mh or complete")

    # TODO: decide whether to use either MH resampling or approximation sampling and
    # remove the alternative, unnecessary complexity in code
    def _resample_auxiliary_transition_variables(
        self, ncores=1, resample_type="mh", use_approximation=True
    ):
        # standard process uses typical list comprehension
        if ncores < 2:
            self.auxiliary_transition_variables = {
                s1: {
                    s2: HDPHMM._resample_auxiliary_transition_atom(
                        (s1, s2),
                        alpha=self.hyperparameters["alpha"],
                        beta=self.beta_transition,
                        n_initial=self.n_initial,
                        n_transition=self.n_transition,
                        auxiliary_transition_variables=self.auxiliary_transition_variables,
                        resample_type=resample_type,
                        use_approximation=use_approximation,
                    )
                    for s2 in self.states
                }
                for s1 in self.states
            }

        # parallel process uses anonymous functions and mapping
        else:
            # specify ordering of states
            state_pairs = [(s1, s2) for s1 in self.states for s2 in self.states]

            # parallel process resamples
            resample_partial = functools.partial(
                HDPHMM._resample_auxiliary_transition_atom,
                alpha=self.hyperparameters["alpha"],
                beta=self.beta_transition,
                n_initial=self.n_initial,
                n_transition=self.n_transition,
                auxiliary_transition_variables=self.auxiliary_transition_variables,
                resample_type=resample_type,
                use_approximation=use_approximation,
            )
            pool = multiprocessing.Pool(processes=ncores)
            auxiliary_transition_variables = pool.map(resample_partial, state_pairs)
            pool.close()

            # store as dictionary
            for pair_n in range(len(state_pairs)):
                state1, state2 = state_pairs[pair_n]
                self.auxiliary_transition_variables[state1][
                    state2
                ] = auxiliary_transition_variables[pair_n]

    def _get_beta_transition_metaparameters(self):
        """
        Calculate parameters for the Dirichlet posterior of the transition beta
        variables (with infinite states aggregated into 'None' state)
        :return: dict, with a key for each state and None, and values equal to parameter
        values
        """
        # aggregate
        params = {
            s2: sum(self.auxiliary_transition_variables[s1][s2] for s1 in self.states)
            for s2 in self.states
        }
        params[None] = self.hyperparameters["gamma"]
        return params

    def resample_beta_transition(
        self, ncores=1, auxiliary_resample_type="mh", use_approximation=True, eps=1e-12
    ):
        """
        Resample the beta values used to calculate the starting and transition
        probabilities.
        :param ncores: int. Number of cores to use in multithreading. Values below 2
        mean the resampling step is not parallelised.
        :param auxiliary_resample_type: either "mh" or "complete". Impacts the way
        in which the auxiliary transition variables are estimated.
        :param use_approximation: bool, flag to indicate whether an approximate
        resampling should occur. ignored if `auxiliary_resample_type` is "mh"
        :param eps: shrinkage parameter to avoid rounding error.
        :return: None
        """
        # auxiliary variables must be resampled to resample beta variables
        self._resample_auxiliary_transition_variables(
            ncores=ncores,
            resample_type=auxiliary_resample_type,
            use_approximation=use_approximation,
        )

        # resample from Dirichlet posterior
        params = self._get_beta_transition_metaparameters()
        temp_result = np.random.dirichlet(list(params.values())).tolist()
        beta_transition = dict(zip(list(params.keys()), temp_result))
        self.beta_transition = shrink_probabilities(beta_transition, eps)

    def calculate_beta_transition_loglikelihood(self):
        # get Dirichlet hyperparameters
        params = self._get_beta_transition_metaparameters()
        ll_transition = np.log(
            stats.dirichlet.pdf(
                [self.beta_transition[s] for s in params.keys()],
                [params[s] for s in params.keys()],
            )
        )
        return ll_transition

    def _get_beta_emission_metaparameters(self):
        """
        Calculate parameters for the Dirichlet posterior of the emission beta variables
        (with infinite states aggregated into 'None' state)
        :return: dict, with a key for each emission, and values equal to parameter
        values
        """
        # aggregate
        params = {
            e: sum(self.n_emission[s][e] for s in self.states)
            + self.hyperparameters["gamma_emission"] / self.n
            for e in self.emissions
        }
        return params

    def resample_beta_emission(self, eps=1e-12):
        """
        Resample the beta values used to calculate the emission probabilties.
        :param eps: Minimum value for expected value before resampling.
        :return: None.
        """
        # resample from Dirichlet posterior
        params = self._get_beta_emission_metaparameters()
        temp_result = np.random.dirichlet(list(params.values())).tolist()
        beta_emission = dict(zip(list(params.keys()), temp_result))
        self.beta_emission = shrink_probabilities(beta_emission, eps)

    def calculate_beta_emission_loglikelihood(self):
        # get Dirichlet hyperparameters
        params = self._get_beta_emission_metaparameters()
        ll_emission = np.log(
            stats.dirichlet.pdf(
                [self.beta_emission[e] for e in self.emissions],
                [params[e] for e in self.emissions],
            )
        )
        return ll_emission

    def _get_p_initial_metaparameters(self):
        params = {
            s: self.n_initial[s]
            + self.hyperparameters["alpha"] * self.beta_transition[s]
            for s in self.states
        }
        params[None] = self.hyperparameters["alpha"] * self.beta_transition[None]
        return params

    def resample_p_initial(self, eps=1e-12):
        """
        Resample the starting probabilities. Performed as a sample from the posterior
        distribution, which is a Dirichlet with pseudocounts and actual counts combined.
        :param eps: minimum expected value.
        :return: None.
        """
        params = self._get_p_initial_metaparameters()
        temp_result = np.random.dirichlet(list(params.values())).tolist()
        p_initial = dict(zip(list(params.keys()), temp_result))
        self.p_initial = shrink_probabilities(p_initial, eps)

    def calculate_p_initial_loglikelihood(self):
        params = self._get_p_initial_metaparameters()
        ll_initial = np.log(
            stats.dirichlet.pdf(
                [self.p_initial[s] for s in params.keys()],
                [params[s] for s in params.keys()],
            )
        )
        return ll_initial

    def _get_p_transition_metaparameters(self, state):
        if self.sticky:
            params = {
                s2: self.n_transition[state][s2]
                + self.hyperparameters["alpha"]
                * (1 - self.hyperparameters["kappa"])
                * self.beta_transition[s2]
                for s2 in self.states
            }
            params[None] = (
                self.hyperparameters["alpha"]
                * (1 - self.hyperparameters["kappa"])
                * self.beta_transition[None]
            )
            params[state] += (
                self.hyperparameters["alpha"] * self.hyperparameters["kappa"]
            )
        else:
            params = {
                s2: self.n_transition[state][s2]
                + self.hyperparameters["alpha"] * self.beta_transition[s2]
                for s2 in self.states
            }
            params[None] = self.hyperparameters["alpha"] * self.beta_transition[None]

        return params

    def resample_p_transition(self, eps=1e-12):
        """
        Resample the transition probabilities from the current beta values and kappa
        value, if the chain is sticky.
        :param eps: minimum expected value passed to Dirichlet sampling step
        :return: None
        """
        # empty current transition values
        self.p_transition = {}

        # refresh each state in turn
        for state in self.states:
            params = self._get_p_transition_metaparameters(state)
            temp_result = np.random.dirichlet(list(params.values())).tolist()
            p_transition_state = dict(zip(list(params.keys()), temp_result))
            self.p_transition[state] = shrink_probabilities(p_transition_state, eps)

        # add transition probabilities from unseen states
        # note: no stickiness update because these are aggregated states
        params = {
            k: self.hyperparameters["alpha"] * v
            for k, v in self.beta_transition.items()
        }
        temp_result = np.random.dirichlet(list(params.values())).tolist()
        p_transition_none = dict(zip(list(params.keys()), temp_result))
        self.p_transition[None] = shrink_probabilities(p_transition_none, eps)

    def calculate_p_transition_loglikelihood(self):
        """
        Note: this calculates the likelihood over all entries in the transition matrix.
        If chains have been resampled (this is the case during MCMC sampling, for
        example), then there may be entries in the transition matrix that no longer
        correspond to actual states.
        :return:
        """
        ll_transition = 0
        states = self.p_transition.keys()

        # get probability for each state
        for state in states:
            params = self._get_p_transition_metaparameters(state)
            ll_transition += np.log(
                stats.dirichlet.pdf(
                    [self.p_transition[state][s] for s in states],
                    [params[s] for s in states],
                )
            )

        # get probability for aggregate state
        params = {
            k: self.hyperparameters["alpha"] * v
            for k, v in self.beta_transition.items()
        }
        ll_transition += np.log(
            stats.dirichlet.pdf(
                [self.p_transition[None][s] for s in states],
                [params[s] for s in states],
            )
        )

        return ll_transition

    def _get_p_emission_metaparameters(self, state):
        params = {
            e: self.n_emission[state][e]
            + self.hyperparameters["alpha_emission"] * self.beta_emission[e]
            for e in self.emissions
        }
        return params

    def resample_p_emission(self, eps=1e-12):
        """
        resample emission parameters from emission priors and counts.
        :param eps: minimum expected value passed to Dirichlet distribution
        :return: None
        """
        # find hyperparameters
        for state in self.states:
            params = self._get_p_emission_metaparameters(state)
            temp_result = np.random.dirichlet(list(params.values())).tolist()
            p_emission_state = dict(zip(list(params.keys()), temp_result))
            self.p_emission[state] = shrink_probabilities(p_emission_state, eps)

        # add emission probabilities from unseen states
        params = {
            k: self.hyperparameters["alpha_emission"] * v
            for k, v in self.beta_emission.items()
        }
        temp_result = np.random.dirichlet(list(params.values())).tolist()
        p_emission_none = dict(zip(list(params.keys()), temp_result))
        self.p_emission[None] = shrink_probabilities(p_emission_none, eps)

    def calculate_p_emission_loglikelihood(self):
        ll_emission = 0

        # get probability for each state
        for state in self.states:
            params = self._get_p_emission_metaparameters(state)
            ll_emission += np.log(
                stats.dirichlet.pdf(
                    [self.p_emission[state][e] for e in self.emissions],
                    [params[e] for e in self.emissions],
                )
            )

        # get probability for aggregate state
        params = {
            k: self.hyperparameters["alpha_emission"] * v
            for k, v in self.beta_emission.items()
        }
        ll_emission += np.log(
            stats.dirichlet.pdf(
                [self.p_emission[None][e] for e in self.emissions],
                [params[e] for e in self.emissions],
            )
        )

        return ll_emission

    def print_fit_parameters(self):
        """
        Prints a copy of the current state counts.
        Used for convenient checking in a command line environment.
        For dictionaries containing the raw values, use the `n_*` attributes.
        :return:
        """
        # create copies to avoid editing
        n_initial = copy.deepcopy(self.n_initial)
        n_emission = copy.deepcopy(self.n_emission)
        n_transition = copy.deepcopy(self.n_transition)

        # make nested lists for clean printing
        initial = [[str(s)] + [str(n_initial[s])] for s in self.states]
        initial.insert(0, ["S_i", "Y_0"])
        emissions = [
            [str(s)] + [str(n_emission[s][e]) for e in self.emissions]
            for s in self.states
        ]
        emissions.insert(0, ["S_i \\ E_i"] + list(map(str, self.emissions)))
        transitions = [
            [str(s1)] + [str(n_transition[s1][s2]) for s2 in self.states]
            for s1 in self.states
        ]
        transitions.insert(0, ["S_i \\ S_j"] + list(map(lambda x: str(x), self.states)))

        # format tables
        ti = terminaltables.DoubleTable(initial, "Starting state counts")
        te = terminaltables.DoubleTable(emissions, "Emission counts")
        tt = terminaltables.DoubleTable(transitions, "Transition counts")
        ti.padding_left = 1
        ti.padding_right = 1
        te.padding_left = 1
        te.padding_right = 1
        tt.padding_left = 1
        tt.padding_right = 1
        ti.justify_columns[0] = "right"
        te.justify_columns[0] = "right"
        tt.justify_columns[0] = "right"

        # print tables
        print("\n")
        print(ti.table)
        print("\n")
        print(te.table)
        print("\n")
        print(tt.table)
        print("\n")

        #
        return None

    def print_probabilities(self):
        """
        Prints a copy of the current probabilities.
        Used for convenient checking in a command line environment.
        For dictionaries containing the raw values, use the `p_*` attributes.
        :return:
        """
        # create copies to avoid editing
        p_initial = copy.deepcopy(self.p_initial)
        p_emission = copy.deepcopy(self.p_emission)
        p_transition = copy.deepcopy(self.p_transition)

        # convert to nested lists for clean printing
        p_initial = [[str(s)] + [str(round(p_initial[s], 3))] for s in self.states]
        p_emission = [
            [str(s)] + [str(round(p_emission[s][e], 3)) for e in self.emissions]
            for s in self.states
        ]
        p_transition = [
            [str(s1)] + [str(round(p_transition[s1][s2], 3)) for s2 in self.states]
            for s1 in self.states
        ]
        p_initial.insert(0, ["S_i", "Y_0"])
        p_emission.insert(0, ["S_i \\ E_j"] + [str(e) for e in self.emissions])
        p_transition.insert(0, ["S_i \\ E_j"] + [str(s) for s in self.states])

        # format tables
        ti = terminaltables.DoubleTable(p_initial, "Starting state probabilities")
        te = terminaltables.DoubleTable(p_emission, "Emission probabilities")
        tt = terminaltables.DoubleTable(p_transition, "Transition probabilities")
        te.padding_left = 1
        te.padding_right = 1
        tt.padding_left = 1
        tt.padding_right = 1
        te.justify_columns[0] = "right"
        tt.justify_columns[0] = "right"

        # print tables
        print("\n")
        print(ti.table)
        print("\n")
        print(te.table)
        print("\n")
        print(tt.table)
        print("\n")

        #
        return None

    def calculate_chain_loglikelihood(self):
        """
        Calculate the negative log likelihood of the chain, given its current
        latent states. This is calculated based on the observed emission sequences only,
        and not on the probabilities of the hyperparameters.
        :return:
        """
        return sum(
            chain.neglogp_chain(self.p_initial, self.p_emission, self.p_transition)
            for chain in self.chains
        )

    def calculate_loglikelihood(self):
        """
        Negative log-likelihood of the entire HDPHMM object. Combines the likelihoods of
        the transition and emission beta parameters, and of the chains themselves.
        Does not include the probabilities of the hyperparameter priors.
        :return: non-negative float
        """
        return (
            self.calculate_beta_transition_loglikelihood()
            + self.calculate_beta_emission_loglikelihood()
            + self.calculate_p_initial_loglikelihood()
            + self.calculate_p_transition_loglikelihood()
            + self.calculate_p_emission_loglikelihood()
            + self.calculate_chain_loglikelihood()
        )

    def resample_chains(self, ncores=1):
        """
        Resample the latent states in all chains. This uses Beam sampling to improve the
        resampling time.
        :param ncores: int, number of threads to use in multithreading.
        :return: None
        """
        # extract probabilities
        p_initial, p_emission, p_transition = (
            self.p_initial,
            self.p_emission,
            self.p_transition,
        )

        # create temporary function for mapping
        resample_partial = functools.partial(
            Chain.resample_latent_sequence,
            states=list(self.states) + [None],
            p_initial=copy.deepcopy(p_initial),
            p_emission=copy.deepcopy(p_emission),
            p_transition=copy.deepcopy(p_transition),
        )

        # parallel process resamples
        pool = multiprocessing.Pool(processes=ncores)
        new_latent_sequences = pool.map(
            resample_partial,
            ((chain.emission_sequence, chain.latent_sequence) for chain in self.chains),
        )
        pool.close()

        # assign returned latent sequences back to Chains
        for i in range(self.c):
            self.chains[i].latent_sequence = new_latent_sequences[i]

        # update chains using results
        # TODO: parameter check if we should be using alpha or gamma as parameter
        state_generator = dirichlet_process_generator(
            self.hyperparameters["gamma"], output_generator=self.state_generator()
        )
        for chain in self.chains:
            chain.latent_sequence = [
                s if s is not None else next(state_generator)
                for s in chain.latent_sequence
            ]

        # update counts
        self._n_update()

    def maximise_hyperparameters(self):
        """
        Choose the MAP (maximum a posteriori) value for the hyperparameters.
        Not yet implemented.
        :return: None
        """
        raise NotImplementedError(
            "This has not yet been written!"
            + " Ping the author if you want it to happen."
        )
        pass

    def resample_hyperparameters(self):
        """
        Resample hyperparameters using a Metropolis Hastings algorithm. Uses a
        straightforward resampling approach, which (for each hyperparameter) samples a
        proposed value according to the prior distribution, and accepts the proposed
        value with probability scaled by the relative probabilities of the model under
        the current and proposed model.
        :return: None
        """
        # iterate and accept each in order
        for param_name in self.priors.keys():
            # don't update kappa if not a sticky chain
            if param_name == "kappa" and not self.sticky:
                continue

            # get current negative log likelihood
            likelihood_curr = self.calculate_loglikelihood()

            # log-likelihood under new parameter value
            param_current = self.hyperparameters[param_name]
            self.hyperparameters[param_name] = self.priors[param_name]()
            likelihood_proposed = self.calculate_loglikelihood()

            # find Metropolis Hasting acceptance probability
            p_accept = min(1, np.exp(likelihood_proposed - likelihood_curr))

            # choose whether to accept
            alpha_accepted = bool(np.random.binomial(n=1, p=p_accept))

            # if we do not accept, revert to the previous value
            if not alpha_accepted:
                self.hyperparameters[param_name] = param_current

    def mcmc(self, n=1000, burn_in=500, save_every=10, ncores=1, verbose=True):
        """
        Use Markov chain Monte Carlo to estimate the starting, transition, and emission
        parameters of the HDPHMM, as well as the number of latent states.
        :param n: int, number of iterations to complete.
        :param burn_in: int, number of iterations to complete before savings results.
        :param save_every: int, only iterations which are a multiple of `save_every`
        will have their results appended to the results.
        :param ncores: int, number of cores to use in multithreaded latent state
        resampling.
        :param verbose: bool, flag to indicate whether iteration-level statistics should
        be printed.
        :return: A dict containing results from every saved iteration. Includes:
          + the number of states of the HDPHMM
          + the negative log likelihood of the entire model
          + the negative log likelihood of the chains only
          + the hyperparameters of the HDPHMM
          + the emission beta values
          + the transition beta values
          + all probability dictionary objects
        """
        # store hyperparameters in a single dict
        results = {
            "state_count": list(),
            "loglikelihood": list(),
            "chain_loglikelihood": list(),
            "hyperparameters": list(),
            "beta_emission": list(),
            "beta_transition": list(),
            "parameters": list(),
        }

        for i in tqdm.tqdm(range(n)):
            # update statistics
            states_prev = copy.copy(self.states)

            # work down hierarchy when resampling
            self.update_states()
            self.resample_hyperparameters()
            self.resample_beta_transition(ncores=ncores)
            self.resample_beta_emission()
            self.resample_p_initial()
            self.resample_p_transition()
            self.resample_p_emission()
            self.resample_chains(ncores=ncores)

            # update computation-heavy statistics
            likelihood_curr = self.calculate_loglikelihood()

            # print iteration summary if required
            if verbose:
                if i == burn_in:
                    tqdm.tqdm.write("Burn-in period complete")
                states_taken = states_prev - self.states
                states_added = self.states - states_prev
                msg = [
                    "Iter: {}".format(i),
                    "Likelihood: {0:.1f}".format(likelihood_curr),
                    "states: {}".format(len(self.states)),
                ]
                if len(states_added) > 0:
                    msg.append("states added: {}".format(states_added))
                if len(states_taken) > 0:
                    msg.append("states removed: {}".format(states_taken))
                tqdm.tqdm.write(", ".join(msg))

            # store results
            if i >= burn_in and i % save_every == 0:
                # get hyperparameters as nested lists
                p_initial = copy.deepcopy(self.p_initial)
                p_emission = copy.deepcopy(self.p_emission)
                p_transition = copy.deepcopy(self.p_transition)

                # save new data
                results["state_count"].append(self.k)
                results["loglikelihood"].append(likelihood_curr)
                results["chain_loglikelihood"].append(
                    self.calculate_chain_loglikelihood()
                )
                results["hyperparameters"].append(copy.deepcopy(self.hyperparameters))
                results["beta_emission"].append(self.beta_emission)
                results["beta_transition"].append(self.beta_transition)
                results["parameters"].append(
                    {
                        "p_initial": p_initial,
                        "p_emission": p_emission,
                        "p_transition": p_transition,
                    }
                )

        # return saved observations
        return results
