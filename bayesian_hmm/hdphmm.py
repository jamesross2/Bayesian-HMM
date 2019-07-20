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

import numpy as np
import random
import copy
import terminaltables
import tqdm
import functools
import multiprocessing
import itertools
import string
from scipy import special
from sympy.functions.combinatorial.numbers import stirling
from .chain import Chain
from warnings import catch_warnings


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


class HDPHMM(object):
    """
    The Hierarchical Dirichlet Process Hidden Markov Model object. In fact, this is a
    sticky-HDPHMM, since we allow a biased self-transition probability.
    """

    def __init__(self, emission_sequences, emissions=None, sticky=True, priors=None):
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

        # adjust kappa prior for non-sticky chains
        if not self.sticky:
            self.priors["kappa"] = lambda: 0
            if priors is not None and "kappa" in priors:
                raise ValueError("`sticky` is False, but non-zero kappa prior given")

        # store initial hyperparameter values
        self.parameters = {param: prior() for param, prior in self.priors.items()}

        # use internal properties to store fit parameters
        self.n_initial = {None: 0}
        self.n_emission = {None: {None: 0}}
        self.n_transition = {None: {None: 0}}

        # use internal properties to store current state for probabilities
        self.p_initial = {None: 1}
        self.p_emission = {None: {None: 1}}
        self.p_transition = {None: {None: 1}}

        # store derived hyperparameters
        self.auxiliary_transition_variables = {None: {None: 0}}
        self.beta_transition = {None: 1}
        self.beta_emission = {None: 1}

        # states & emissions
        if emissions is None:
            emissions = functools.reduce(
                set.union, (set(c.emission_sequence) for c in self.chains), set()
            )
        elif type(emissions) is not set:
            raise ValueError("emissions must be a set")
        self.emissions = emissions
        self.states = set()

        # generate non-repeating character labels for latent states
        self._label_generator = label_generator(string.ascii_lowercase)

        # keep flag to track initialisation
        self._initialised = False

    @property
    def initialised(self):
        """
        Test whether a HDPHMM is initialised.
        :return: bool
        """
        return self._initialised

    @initialised.setter
    def initialised(self, value):
        if value:
            raise AssertionError("HDPHMM must be initialised through initialise method")
        elif not value:
            self._initialised = False
        else:
            raise ValueError("initialised flag must be Boolean")

    @property
    def c(self):
        """
        Number of chains in the HMM.
        :return: int
        """
        return len(self.chains)

    @property
    def k(self):
        """
        Number of latent states in the HMM currently.
        :return: int
        """
        return len(self.states)

    @property
    def n(self):
        """
        Number of unique emissions. If `emissions` was specified when the HDPHMM was
        created, then this counts the number of elements in `emissions`. Otherwise,
        counts the number of observed emissions across all emission sequences.
        :return: int
        """
        return len(self.emissions)

    def tabulate(self):
        """
        Convert the latent and emission sequences for all chains into a single numpy
        array. Array contains an index, which matches a Chain's position in
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

    def __repr__(self):
        return "<bayesian_hmm.HDPHMM, size {C}>".format(C=self.c)

    def __str__(self, print_len=15):
        fs = (
            "bayesian_hmm.HDPHMM,"
            + " ({C} chains, {K} states, {N} emissions, {Ob} observations)"
        )
        return fs.format(C=self.c, K=self.k, N=self.n, Ob=sum(c.T for c in self.chains))

    def state_generator(self):
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
            self.n_transition[label] = {s: 0 for s in self.states.union({label})}
            [self.n_transition[s].update({label: 0}) for s in self.states]
            self.n_emission[label] = {e: 0 for e in self.emissions}

            # update auxiliary transition variables
            self.auxiliary_transition_variables[label] = {
                s2: 1 for s2 in list(self.states) + [label]
            }
            for s1 in self.states:
                self.auxiliary_transition_variables[s1][label] = 1

            # update beta_transition value and split out from current pseudo state
            temp_beta = np.random.beta(1, self.parameters["gamma"])
            self.beta_transition[label] = temp_beta * self.beta_transition[None]
            self.beta_transition[None] = (1 - temp_beta) * self.beta_transition[None]

            # update starting probability
            temp_p_initial = np.random.beta(1, self.parameters["gamma"])
            self.p_initial[label] = temp_p_initial * self.p_initial[None]
            self.p_initial[None] = (1 - temp_p_initial) * self.p_initial[None]

            # update transition from new state
            temp_p_transition = np.random.dirichlet(
                [self.beta_transition[s] for s in list(self.states) + [label, None]]
            )
            self.p_transition[label] = dict(
                zip(list(self.states) + [label, None], temp_p_transition)
            )

            # update transitions into new state
            for state in self.states.union({None}):
                # (note that label not included in self.states)
                temp_p_transition = np.random.beta(1, self.parameters["gamma"])
                self.p_transition[state][label] = (
                    self.p_transition[state][None] * temp_p_transition
                )
                self.p_transition[state][None] = self.p_transition[state][None] * (
                    1 - temp_p_transition
                )

            # update emission probabilities
            temp_p_emission = np.random.dirichlet(
                [
                    self.parameters["alpha"] * self.beta_emission[e]
                    for e in self.emissions
                ]
            )
            self.p_emission[label] = dict(zip(self.emissions, temp_p_emission))

            # save label
            self.states = self.states.union({label})

            #
            yield label

    def initialise(self, k=20):
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
        self.parameters = {param: prior() for param, prior in self.priors.items()}

        # initialise chains
        [c.initialise(states) for c in self.chains]

        # initialise hierarchical priors
        temp_beta = sorted(
            np.random.dirichlet(
                [self.parameters["gamma"] / (self.k + 1)] * (self.k + 1)
            )
        )
        self.beta_transition = dict(zip(list(self.states) + [None], temp_beta))
        self.auxiliary_transition_variables = {
            s1: {s2: 1 for s2 in self.states} for s1 in self.states
        }

        # update counts before resampling
        self._n_update()

        # resample remaining hyperparameters
        self._resample_auxiliary_transition_variables()
        self.resample_beta_emission()
        self.resample_p_initial()
        self.resample_p_transition()
        self.resample_p_emission()

        # set initialised flag
        self._initialised = True

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

        # make sure that we have captured states properly
        states = sorted(
            functools.reduce(
                set.union, (set(c.latent_sequence) for c in self.chains), set()
            )
        )
        self.states = set(states)

        # transition count for non-oracle transitions
        n_initial = {s: 0 for s in self.states}
        n_emission = {s: {e: 0 for e in self.emissions} for s in self.states}
        n_transition = {s1: {s2: 0 for s2 in self.states} for s1 in self.states}

        # increment all relevant parameters while looping over sequence
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

        # store recalculated fit parameters
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
        return m

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
                        self.parameters["alpha"],
                        self.beta_transition,
                        self.n_initial,
                        self.n_transition,
                        self.auxiliary_transition_variables,
                        resample_type,
                        use_approximation,
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
                alpha=self.parameters["alpha"],
                beta=self.beta_transition,
                n_initial=self.n_initial,
                n_transition=self.n_transition,
                auxiliary_transition_variables=self.auxiliary_transition_variables,
                resample_type=resample_type,
                use_approximation=use_approximation,
            )

            with multiprocessing.Pool(processes=ncores) as pool:
                auxiliary_transition_variables = pool.map(resample_partial, state_pairs)

            # store as dictionary
            for pair_n in range(len(state_pairs)):
                state1, state2 = state_pairs[pair_n]
                self.auxiliary_transition_variables[state1][
                    state2
                ] = auxiliary_transition_variables[pair_n]

    def resample_beta_transition(
        self, ncores=1, auxiliary_resample_type="mh", use_approximation=True, eps=1e-2
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
        :param eps: error term used in approximations to avoid numerical roundoff errors
        :return: None
        """

        # resample auxiliary variables
        self._resample_auxiliary_transition_variables(
            ncores=ncores,
            resample_type=auxiliary_resample_type,
            use_approximation=use_approximation,
        )

        # aggregate
        aggregate_auxiliary_variables = {
            s2: sum(self.auxiliary_transition_variables[s1][s2] for s1 in self.states)
            for s2 in self.states
        }

        # given by auxiliary transition variables plus gamma controlling unseen states
        temp_expected = [aggregate_auxiliary_variables[s2] for s2 in self.states] + [
            self.parameters["gamma"]
        ]
        temp_expected = [max(x, eps) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.beta_transition = dict(zip(list(self.states) + [None], temp_result))

    def resample_beta_emission(self, eps=1e-2):
        """
        Resample the beta values used to calculate the emission probabilties.
        :param eps: Minimum value for expected value before resampling.
        :return: None.
        """
        # given by number of emissions
        temp_expected = [
            sum(self.n_emission[s][e] for s in self.states)
            + self.parameters["gamma_emission"] / self.n
            for e in self.emissions
        ]
        # nake sure no degenerate zeros (some emissions can be unseen permanently)
        temp_expected = [max(x, eps) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.beta_emission = dict(zip(self.emissions, temp_result))

    def resample_p_initial(self, eps=1e-2):
        """
        Resample the starting probabilities. Performed as a sample from the posterior
        distribution, which is a Dirichlet with pseudocounts and actual counts combined.
        :param eps: minimum expected value.
        :return: None.
        """
        # given by hierarchical beta value plus observed starts
        temp_expected = [
            self.n_initial[s2] + self.parameters["alpha"] * self.beta_transition[s2]
            for s2 in self.states
        ]
        temp_expected.append(self.parameters["alpha"] * self.beta_transition[None])
        temp_expected = [max((x, eps)) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.p_initial = {s: p for s, p in zip(list(self.states) + [None], temp_result)}

    def resample_p_transition(self, eps=1e-2):
        """
        Resample the transition probabilities from the current beta values and kappa
        value, if the chain is sticky.
        :param eps: minimum expected value passed to Dirichlet sampling step
        :return: None
        """
        # given by hierarchical beta value plus observed transitions
        for s1 in self.states:
            if self.sticky:
                temp_expected = [
                    self.n_transition[s1][s2]
                    + self.parameters["alpha"]
                    * (1 - self.parameters["kappa"])
                    * self.beta_transition[s2]
                    + (
                        self.parameters["alpha"] * self.parameters["kappa"]
                        if s1 == s2
                        else 0
                    )
                    for s2 in self.states
                ]
                temp_expected.append(
                    self.parameters["alpha"]
                    * (1 - self.parameters["kappa"])
                    * self.beta_transition[None]
                )
            else:
                temp_expected = [
                    self.n_transition[s1][s2]
                    + self.parameters["alpha"] * self.beta_transition[s2]
                    for s2 in self.states
                ]
                temp_expected.append(
                    self.parameters["alpha"] * self.beta_transition[None]
                )
            temp_expected = [max(x, eps) for x in temp_expected]
            temp_result = np.random.dirichlet(temp_expected).tolist()
            self.p_transition[s1] = {
                s2: p for s2, p in zip(list(self.states) + [None], temp_result)
            }

        # add transition probabilities from unseen states
        # note: no stickiness update because these are aggregated states
        temp_expected = [
            self.parameters["alpha"] * b for b in self.beta_transition.values()
        ]
        temp_expected = [max((x, eps)) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.p_transition[None] = {
            s2: p for s2, p in zip(list(self.states) + [None], temp_result)
        }

    def resample_p_emission(self, eps=1e-2):
        """
        resample emission parameters from emission priors and counts.
        :param eps: minimum expected value passed to Dirichlet distribution
        :return: None
        """
        # find parameters
        for s in self.states:
            temp_expected = [
                self.n_emission[s][e]
                + self.parameters["alpha_emission"] * self.beta_emission[e]
                for e in self.emissions
            ]
            temp_expected = [max((x, eps)) for x in temp_expected]
            temp_result = np.random.dirichlet(temp_expected).tolist()
            # update stored transition probability
            self.p_emission[s] = {e: p for e, p in zip(self.emissions, temp_result)}

        # add emission probabilities from unseen states
        temp_expected = [
            self.parameters["alpha_emission"] * b for b in self.beta_emission.values()
        ]
        temp_expected = [max((x, eps)) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.p_emission[None] = {e: p for e, p in zip(self.emissions, temp_result)}

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

    def neglogp_sequences(self):
        """
        Calculate the negative log likelihood of the chain, given its current
        latent states. This is calculated based on the observed emission sequences only,
        and not on the probabilities of the hyperparameters.
        :return:
        """
        return sum(
            chain.neglogp_sequence(self.p_initial, self.p_emission, self.p_transition)
            for chain in self.chains
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
        with multiprocessing.Pool(processes=ncores) as pool:
            new_latent_sequences = pool.map(
                resample_partial,
                (
                    (chain.emission_sequence, chain.latent_sequence)
                    for chain in self.chains
                ),
            )

        # assign returned latent sequences back to Chains
        for i in range(self.c):
            self.chains[i].latent_sequence = new_latent_sequences[i]

        # update chains using results
        # TODO: parameter check if we should be using alpha or gamma as parameter
        state_generator = dirichlet_process_generator(
            self.parameters["gamma"], output_generator=self.state_generator()
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
            neglogp_current = self.neglogp_sequences()

            # log-likelihood under new parameter value
            param_current = self.parameters[param_name]
            self.parameters[param_name] = self.priors[param_name]()
            neglogp_proposed = self.neglogp_sequences()

            # find Metropolis Hasting acceptance probability
            p_accept = min(1, np.exp(-neglogp_proposed + neglogp_current))

            # choose whether to accept
            alpha_accepted = bool(np.random.binomial(n=1, p=p_accept))

            # if we do not accept, revert to the previous value
            if not alpha_accepted:
                self.parameters[param_name] = param_current

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
        :return: A dict containing results from every saved iteration. Each entry
        contains:
          + the number of states of the HDPHMM
          + the negative log likelihood of the emissions
          + the hyperparameters of the HDPHMM
          + the emission beta values
          + the transition beta values
          + all probability dictionary objects
        """
        # store parameters
        state_count_hist = list()
        sequence_prob_hist = list()
        hyperparameter_hist = list()
        beta_emission_hist = list()
        beta_transition_hist = list()
        parameter_hist = list()

        for i in tqdm.tqdm(range(n)):
            # update statistics
            states_prev = copy.copy(self.states)

            # work down hierarchy when resampling
            self.resample_beta_transition(ncores=1)
            self.resample_beta_emission()
            self.resample_p_initial()
            self.resample_p_transition()
            self.resample_p_emission()
            self.resample_hyperparameters()
            self.resample_chains(ncores=ncores)

            # print iteration summary if required
            if verbose:
                if i == burn_in:
                    tqdm.tqdm.write("Burn-in period complete")
                states_taken = states_prev - self.states
                states_added = self.states - states_prev
                msg = [
                    "Iter: {}".format(i),
                    "Likelihood: {0:.1f}".format(self.neglogp_sequences()),
                    "states: {}".format(len(self.states)),
                ]
                if len(states_added) > 0:
                    msg.append("states added: {}".format(states_added))
                if len(states_taken) > 0:
                    msg.append("states removed: {}".format(states_taken))
                tqdm.tqdm.write(", ".join(msg))

            # store results
            if i >= burn_in and i % save_every == 0:
                # get parameters as nested lists
                p_initial = copy.deepcopy(self.p_initial)
                p_emission = copy.deepcopy(self.p_emission)
                p_transition = copy.deepcopy(self.p_transition)

                # save new data
                state_count_hist.append(self.k)
                sequence_prob_hist.append(self.neglogp_sequences())
                hyperparameter_hist.append(copy.deepcopy(self.parameters))
                beta_emission_hist.append(self.beta_emission)
                beta_transition_hist.append(self.beta_transition)
                parameter_hist.append(
                    {
                        "p_initial": p_initial,
                        "p_emission": p_emission,
                        "p_transition": p_transition,
                    }
                )

        # return saved observations
        return {
            "state_count": state_count_hist,
            "chain_neglogp": sequence_prob_hist,
            "hyperparameters": hyperparameter_hist,
            "beta_emission": beta_emission_hist,
            "beta_transition": beta_transition_hist,
            "parameters": parameter_hist,
        }
