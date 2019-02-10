"""
Hierarchical Dirichlet Process Hidden Markov Model
2019-01-02
"""

# standard imports
import numpy as np
import pandas as pd
import random
from scipy import special
import copy
from terminaltables import DoubleTable  # use this to print parameter matrices for humans
from tqdm import tqdm  # add a progress bar to Gibbs sampling
from functools import reduce, partial
import multiprocessing
from sympy.functions.combinatorial.numbers import stirling  # stirling numbers
import itertools  # for infinite generators


# used to give human-friendly labels to states as they are created
def label_generator(labels):
    """
    Returns a generator object which yields unique labels of the form
    a, b, ..., z, a1, b1, ...
    """
    x, y, z = 0, 0, ''
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
    Returns a generator object which yields subsequent draws from a single dirichlet process
    """
    if output_generator is None:
        output_generator = itertools.count(start=0, step=1)
    sequence = []
    while True:
        if random.uniform(0, 1) > (len(sequence) / (len(sequence) + alpha)):
            val = output_generator()
        else:
            val = random.choice(sequence)
        sequence.append(val)
        yield val


# Chain stores a single markov emisison sequence plus associated latent variables
class Chain(object):
    def __init__(self, sequence):

        # initialise & store sequences
        self.emission_sequence = copy.deepcopy(sequence)
        self.latent_sequence = [None for _ in sequence]
        self.auxiliary_beam_variables = [None for _ in sequence]

        # calculate dependent hyperparameters
        self.T = len(self.emission_sequence)

        # keep flag to track initialisation
        self._initialised_flag = False

    def __len__(self):
        return self.T

    @property
    def initialised_flag(self):
        return self._initialised_flag

    @initialised_flag.setter
    def initialised_flag(self, value):
        if value is True:
            raise AssertionError("Chain must be initialised through initialise_chain method")
        elif value is False:
            self._initialised_flag = False
        else:
            raise ValueError("initialised flag must be Boolean")

    # format for non-string printing
    def __repr__(self):
        return '<Chain size {0}>'.format(len(self.emission_sequence))

    # format for conversion to string
    def __str__(self, print_len=15):
        print_len = min(print_len-1, self.T-1)
        return 'Chain size={T}, seq={s}'.format(
            T=self.T,
            s=['{s}:{e}'.format(s=s, e=e) for s, e in
               zip(self.latent_sequence, self.emission_sequence)][:print_len] + ['...'])

    # convert latent and observed sequence into more convenient numpy array
    def tabulate(self):
        return np.column_stack((copy.copy(self.latent_sequence), copy.copy(self.emission_sequence)))

    # introduce randomly sampled states for all latent variables in Chain
    def initialise(self, states):
        # create latent sequence
        self.latent_sequence = random.choices(states, k=self.T)

        # update observations
        self.auxiliary_beam_variables = np.random.uniform(0, 1, size=self.T)
        self._initialised_flag = True

    def neglogp_sequence(self, p_initial, p_emission, p_transition):
        # edge case: zero-length sequence
        if self.T == 0:
            return 0

        # get probability of starting state & emission, and all remaining transition & emissions
        # np.prod([])==1, so this is safe
        p_initial = (np.log(p_initial[self.latent_sequence[0]]) +
                     np.log(p_emission[self.latent_sequence[0]][self.emission_sequence[0]]))
        p_remainder = [np.log(p_emission[self.latent_sequence[t]][self.emission_sequence[t]]) +
                       np.log(p_transition[self.latent_sequence[t-1]][self.latent_sequence[t]])
                       for t in range(1, self.T)]

        # take log and sum for result
        return - (p_initial + sum(p_remainder))

    def resample_auxiliary_beam_variables(self, p_initial, p_transition):
        # find transition probabilities first
        temp_p_transition = ([p_initial[self.latent_sequence[0]]] +
                             [p_transition[self.latent_sequence[t]][self.latent_sequence[t+1]]
                              for t in range(self.T-1)])
        # initialise u_t
        self.auxiliary_beam_variables = [np.random.uniform(0, p) for p in temp_p_transition]

    def resample_latent_sequence(self, states, p_initial, p_emission, p_transition):
        # make sure beam variables are looking nice
        self.resample_auxiliary_beam_variables(p_initial, p_transition)

        # adjust latent sequence
        self.latent_sequence = Chain.resample_latent_sequence_static(
            (self.emission_sequence, self.auxiliary_beam_variables),
            states, p_initial, p_emission, p_transition)

    @staticmethod
    def resample_latent_sequence_static(sequences, states, p_initial, p_emission, p_transition):
        # extract size information
        emission_sequence, auxiliary_vars = sequences
        t = len(emission_sequence)

        # edge case: zero-length sequence
        if t == 0:
            return []

        # initialise historical P(s_t | u_{1:t}, y_{1:t}) and latent sequence
        p_history = [None] * t
        latent_sequence = [None]*t

        # compute probability of state t (currently the starting state t==0)
        p_history[0] = {s: p_initial[s] * p_emission[s][emission_sequence[0]] if p_initial[s] > auxiliary_vars[0] else 0
                        for s in states}
        # for remaining states, probabilities are function of emission and transition
        for t in range(1, t):
            p_temp = {s2: sum(p_history[t-1][s1] for s1 in states if p_transition[s1][s2] > auxiliary_vars[t]) *
                      p_emission[s2][emission_sequence[t]] for s2 in states}
            p_temp_total = sum(p_temp.values())
            p_history[t] = {s: p_temp[s] / p_temp_total for s in states}

        # choose ending state
        latent_sequence[t-1] = random.choices(
            tuple(p_history[t-1].keys()),
            weights=tuple(p_history[t-1].values()),
            k=1)[0]

        # work backwards to compute new latent sequence
        for t in range(t-2, -1, -1):
            p_temp = {s1: p_history[t][s1] * p_transition[s1][latent_sequence[t+1]]
                      if p_transition[s1][latent_sequence[t+1]] > auxiliary_vars[t+1] else 0
                      for s1 in states}
            latent_sequence[t] = random.choices(tuple(p_temp.keys()), weights=tuple(p_temp.values()), k=1)[0]

        # latent sequence now completely filled
        return latent_sequence


class HierarchicalDirichletProcessHiddenMarkovModel(object):
    def __init__(self,
                 emission_sequences,
                 emissions=None,
                 sticky=True,
                 alpha_prior=lambda: np.random.gamma(2, 2),
                 gamma_prior=lambda: np.random.gamma(3, 3),
                 alpha_emission_prior=lambda: np.random.gamma(2, 2),
                 gamma_emission_prior=lambda: np.random.gamma(3, 3),
                 kappa_prior=lambda: np.random.beta(1, 1)):
        """
        Create an HDP-HMM object.

        Parameters control the following:
          * alpha: variability in observed transition distributions. higher values of alpha
            keep rows of the transition matrix more similar to the beta parameters
          * gamma: controls the relative weight given to unseen states when estimating beta.
            higher values of gamma mean the chain is more likely to explore new states.
          * alpha_emission: controls how tightly the conditional emission distributions follow their hierarchical prior.
            higher values of alpha_emission mean more strength in the hierarchical prior.
          * gamma_emission: controls the strength of uninformative prior in the emission distribution.
            higher values of gamma mean more strength of belief in the prior.
          * kappa: TBD. will control strength of self-transition.
        """
        # store chains
        self.chains = [Chain(sequence) for sequence in emission_sequences]

        # sticky flag
        if type(sticky) is not bool:
            raise ValueError('`sticky` must be type bool')
        self.sticky = sticky

        # store hyperparameter priors
        self.alpha_prior, self.gamma_prior = alpha_prior, gamma_prior,
        self.alpha_emission_prior, self.gamma_emission_prior = alpha_emission_prior, gamma_emission_prior
        self.kappa_prior = kappa_prior

        # store initial hyperparameter values
        self.alpha = 1
        self.gamma = 1
        self.alpha_emission = 1
        self.gamma_emission = 1
        self.kappa = 0.2 if self.sticky else 0.0

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
            emissions = reduce(set.union, (set(c.emission_sequence) for c in self.chains), set())
        elif type(emissions) is not set:
            raise ValueError("emissions must be a set")
        self.emissions = emissions
        self.states = set()

        # generate non-repeating character labels for latent states
        self._label_generator = label_generator('abcdefghijklmnopqrstuvwxyz')

        # keep flag to track initialisation
        self._initialised = False

    @property
    def initialised(self):
        return self._initialised

    @initialised.setter
    def initialised(self, value):
        if value:
            raise AssertionError("must be initialised through initialise method")
        elif not value:
            self._initialised = False
        else:
            raise ValueError("initialised flag must be Boolean")

    # number of chains
    @property
    def c(self):
        return len(self.chains)

    # number of latent states
    @property
    def k(self):
        return len(self.states)

    # number of emissions observed
    @property
    def n(self):
        return len(self.emissions)

    # convert latent and observed sequence into more convenient numpy array
    def tabulate(self):
        df = pd.concat((c.tabulate() for c in self.chains), axis=0, keys=range(self.c))
        return df

    # string format when returned as object
    def __repr__(self):
        return '<ChainFamily size {C}>'.format(C=self.c)

    # string format when converted to string
    def __str__(self, print_len=15):
        return 'HDP-HMM ({C} chains, {K} states, {N} emissions, {Ob} observations)'.format(
            C=self.c, K=self.k, N=self.n, Ob=sum(c.T for c in self.chains))

    # create a single new state
    def generate_state(self):
        # first, create the new label. then, append to existing properties (beta variables, counts, probabilities).
        # finally, append new label to self.states
        label = next(self._label_generator)

        # update counts (use zeros & assume n_update called to update to actual counts)
        # state irrelevant for constant count (all zeros)
        self.n_initial[label] = 0
        self.n_transition[label] = {s: 0 for s in self.states.union({label})}
        [self.n_transition[s].update({label: 0}) for s in self.states]
        self.n_emission[label] = {e: 0 for e in self.emissions}

        # update auxiliary transition variables
        self.auxiliary_transition_variables[label] = {s2: 1 for s2 in list(self.states)+[label]}
        for s1 in self.states:
            self.auxiliary_transition_variables[s1][label] = 1

        # choose new beta_transition value (chosen as beta random variable), and split out from current pseudo state
        temp_beta = np.random.beta(1, self.gamma)
        self.beta_transition[label] = temp_beta * self.beta_transition[None]
        self.beta_transition[None] = (1-temp_beta) * self.beta_transition[None]

        # update starting probability
        temp_p_initial = np.random.beta(1, self.gamma)
        self.p_initial[label] = temp_p_initial * self.p_initial[None]
        self.p_initial[None] = (1-temp_p_initial) * self.p_initial[None]

        # update transition from new state
        temp_p_transition = np.random.dirichlet([self.beta_transition[s] for s in list(self.states)+[label, None]])
        self.p_transition[label] = dict(zip(list(self.states)+[label, None], temp_p_transition))

        # update transitions into new state (note that label not included in self.states)
        for state in self.states.union({None}):
            temp_p_transition = np.random.beta(1, self.gamma)
            self.p_transition[state][label] = self.p_transition[state][None] * temp_p_transition
            self.p_transition[state][None] = self.p_transition[state][None] * (1-temp_p_transition)

        # update emission probabilities
        temp_p_emission = np.random.dirichlet([self.alpha_emission * self.beta_emission[e] for e in self.emissions])
        self.p_emission[label] = dict(zip(self.emissions, temp_p_emission))

        # save label
        # self.states.update(label)
        # TODO: solve: this line _should_ be the same as the below, but ends up storing state '1' in self.states instead
        self.states = self.states.union({label})

        #
        return label

    # initialise chain
    def initialise(self, k=20):
        # create as many states as needed (do not use generate_state function yet though)
        states = [next(self._label_generator) for _ in range(k)]
        self.states = set(states)

        # set hyperparameters
        self.alpha = self.alpha_prior()
        self.gamma = self.gamma_prior()
        self.alpha_emission = self.alpha_emission_prior()
        self.gamma_emission = self.gamma_emission_prior()
        self.kappa = self.kappa_prior()

        # latent states and transition betas are only parameters required to be initialised, all others can be sampled
        [c.initialise(states) for c in self.chains]
        temp_beta = sorted(np.random.dirichlet([self.gamma / (self.k + 1)] * (self.k + 1)))
        self.beta_transition = dict(zip(list(self.states)+[None], temp_beta))
        self.auxiliary_transition_variables = {s1: {s2: 1 for s2 in self.states} for s1 in self.states}

        # update counts before resampling
        self.n_update()

        # resample remaining hyperparameters
        self.resample_auxiliary_transition_variables()
        self.resample_p_initial()
        self.resample_p_transition()
        self.resample_beta_emission()
        self.resample_p_emission()
        [c.resample_auxiliary_beam_variables(self.p_initial, self.p_transition) for c in self.chains]

    # get global fit parameters
    def n_update(self):
        # check that all chains are initialised
        if any(not chain.initialised_flag for chain in self.chains):
            raise AssertionError("Chains must be initialised before calculating fit parameters")

        # make sure that we have captured states properly
        states = sorted(reduce(set.union, (set(c.latent_sequence) for c in self.chains), set()))
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
            n_emission[chain.latent_sequence[chain.T-1]][chain.emission_sequence[chain.T-1]] += 1

            # increment all transitions and emissions within chain
            for t in range(chain.T - 1):
                n_emission[chain.latent_sequence[t]][chain.emission_sequence[t]] += 1
                n_transition[chain.latent_sequence[t]][chain.latent_sequence[t+1]] += 1

        # store recalculated fit parameters
        self.n_initial = n_initial
        self.n_emission = n_emission
        self.n_transition = n_transition

    @staticmethod
    def _resample_auxiliary_transition_atom_complete(alpha, beta, n, use_approximation=True):
        # only applies resampling for m
        p_required = np.random.uniform(0, 1)
        m = 0
        p_cumulative = 0
        scale = alpha*beta
        # start with precise probabilities, but after one failure use only the approximation
        if not use_approximation:
            try:
                logp_constant = np.log(special.gamma(scale)) - np.log(special.gamma(scale + n))
                while p_cumulative == 0 or p_cumulative < p_required and m < n:
                    # accumulate probability
                    m += 1
                    logp_accept = m * np.log(scale) + np.log(stirling(n, m, kind=1)) + logp_constant
                    p_cumulative += np.exp(logp_accept)
            except (RecursionError, OverflowError):
                # correct for failed case before
                m -= 1
        while p_cumulative < p_required and m < n:
            # problems with stirling recursion (large n & m), use approximation instead
            # magic number is the Euler constant, stirling approximation from Wikipedia, factorial approx from Wikipedia
            # combination reduces (after much algebra) to below (needs some additional approximations)
            m += 1
            logp_accept = (
                m
                + (m + scale - 0.5) * np.log(scale)
                + (m - 1) * np.log(0.57721 + np.log(n-1))
                - (m - 0.5) * np.log(m)
                - scale * np.log(scale + n)
                - scale)
            p_cumulative += np.exp(logp_accept)
        # breaks out of loop after m is sufficiently large
        return m

    @staticmethod
    def _resample_auxiliary_transition_atom_mh(alpha, beta, n, m_curr, use_approximation=True):
        # propose new m
        n = max(n, 1)
        m_proposed = random.choice(range(1, n+1))
        if m_curr > n:
            # n altered since last m, making value degenerate. guaranteed acceptance anyway after log-transformed zeros
            return m_proposed

        # find relative probabilities
        if use_approximation and n > 10:
            logp_diff = (
                (m_proposed-0.5)*np.log(m_proposed)
                - (m_curr-0.5)*np.log(m_curr)
                + (m_proposed - m_curr) * np.log(alpha * beta * np.exp(1))
                + (m_proposed - m_curr) * np.log(0.57721 + np.log(n-1)))
        else:
            p_curr = float(stirling(n, m_curr, kind=1)) * ((alpha*beta) ** m_curr)
            p_proposed = float(stirling(n, m_proposed, kind=1)) * ((alpha*beta) ** m_proposed)
            logp_diff = np.log(p_proposed) - np.log(p_curr)

        # use MH variable to decide whether to accept m_proposed
        p_accept = min(1, np.exp(logp_diff))
        p_accept = bool(np.random.binomial(n=1, p=p_accept))  # convert to boolean
        return m_proposed if p_accept else m_curr

    @staticmethod
    def _resample_auxiliary_transition_atom(
      state_pair,
      alpha,
      beta,
      n_initial,
      n_transition,
      auxiliary_transition_variables,
      resample_type='mh',
      use_approximation=True):
        # extract states
        state1, state2 = state_pair

        # apply resampling
        if resample_type == 'mh':
            return HierarchicalDirichletProcessHiddenMarkovModel._resample_auxiliary_transition_atom_mh(
                alpha,
                beta[state2],
                n_initial[state2] + n_transition[state1][state2],
                auxiliary_transition_variables[state1][state2],
                use_approximation)
        elif resample_type == 'complete':
            return HierarchicalDirichletProcessHiddenMarkovModel._resample_auxiliary_transition_atom_complete(
                alpha, beta[state2], n_initial[state2] + n_transition[state1][state2], use_approximation)
        else:
            raise ValueError('resample_type must be either mh or complete')

    def resample_auxiliary_transition_variables(self, ncores=1, resample_type='mh', use_approximation=True):
        # standard process uses typical list comprehension
        if ncores < 2:
            self.auxiliary_transition_variables = {
                s1: {
                    s2: HierarchicalDirichletProcessHiddenMarkovModel._resample_auxiliary_transition_atom(
                        # TODO: test if we can remove the max input
                        (s1, s2),
                        self.alpha,
                        self.beta_transition,
                        self.n_initial,
                        self.n_transition,
                        self.auxiliary_transition_variables,
                        resample_type,
                        use_approximation)
                    for s2 in self.states}
                for s1 in self.states}

        # parallel process uses anonymous functions and mapping
        else:
            # specify ordering of states
            state_pairs = [(s1, s2) for s1 in self.states for s2 in self.states]

            # parallel process resamples
            resample_partial = partial(
                HierarchicalDirichletProcessHiddenMarkovModel._resample_auxiliary_transition_atom,
                alpha=self.alpha,
                beta=self.beta_transition,
                n_initial=self.n_initial,
                n_transition=self.n_transition,
                auxiliary_transition_variables=self.auxiliary_transition_variables,
                resample_type=resample_type,
                use_approximation=use_approximation)

            with multiprocessing.Pool(processes=ncores) as pool:
                auxiliary_transition_variables = pool.map(resample_partial, state_pairs)

            # store as dictionary
            for pair_n in range(len(state_pairs)):
                state1, state2 = state_pairs[pair_n]
                self.auxiliary_transition_variables[state1][state2] = auxiliary_transition_variables[pair_n]

    def resample_beta_transition(self, ncores=1, auxiliary_resample_type='mh', use_approximation=True, eps=1e-2):
        """
        the `use_approximation` value is ignore if `use_metropolis_hasting` is `True`.
        """
        # resample auxiliary variables
        self.resample_auxiliary_transition_variables(
            ncores=ncores,
            resample_type=auxiliary_resample_type,
            use_approximation=use_approximation)

        # aggregate
        aggregate_auxiliary_variables = {s2: sum(self.auxiliary_transition_variables[s1][s2] for s1 in self.states)
                                         for s2 in self.states}

        # given by auxiliary transition variables plus gamma controlling unseen states
        temp_expected = [aggregate_auxiliary_variables[s2] for s2 in self.states] + [self.gamma]
        temp_expected = [max(x, eps) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.beta_transition = dict(zip(list(self.states)+[None], temp_result))

    def resample_beta_emission(self, eps=1e-2):
        # given by number of emissions
        temp_expected = [sum(self.n_emission[s][e] for s in self.states) +
                         self.gamma_emission / self.n
                         for e in self.emissions]
        # nake sure no degenerate zeros (some emissions can be unseen permanently)
        temp_expected = [max(x, eps) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.beta_emission = dict(zip(self.emissions, temp_result))

    def resample_p_initial(self, eps=1e-2):
        # given by hierarchical beta value plus observed starts
        temp_expected = [self.n_initial[s2] + self.alpha * self.beta_transition[s2] for s2 in self.states]
        temp_expected.append(self.alpha * self.beta_transition[None])
        temp_expected = [max((x, eps)) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.p_initial = {s: p for s, p in zip(list(self.states) + [None], temp_result)}

    def resample_p_transition(self, eps=1e-2):
        # given by hierarchical beta value plus observed transitions
        for s1 in self.states:
            if self.sticky:
                temp_expected = [self.n_transition[s1][s2] + self.alpha * (1-self.kappa) * self.beta_transition[s2] +
                                 (self.alpha * self.kappa if s1 == s2 else 0) for s2 in self.states]
                temp_expected.append(self.alpha * (1-self.kappa) * self.beta_transition[None])
            else:
                temp_expected = [self.n_transition[s1][s2] + self.alpha * self.beta_transition[s2]
                                 for s2 in self.states]
                temp_expected.append(self.alpha * self.beta_transition[None])
            temp_expected = [max(x, eps) for x in temp_expected]
            temp_result = np.random.dirichlet(temp_expected).tolist()
            self.p_transition[s1] = {s2: p for s2, p in zip(list(self.states)+[None], temp_result)}

        # add transition probabilities from unseen states
        # note: no stickiness update because these are aggregated states
        temp_expected = [self.alpha * b for b in self.beta_transition.values()]
        temp_expected = [max((x, eps)) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.p_transition[None] = {s2: p for s2, p in zip(list(self.states)+[None], temp_result)}

    def resample_p_emission(self, eps=1e-2):
        # find parameters
        for s in self.states:
            temp_expected = [self.n_emission[s][e] + self.alpha_emission * self.beta_emission[e]
                             for e in self.emissions]
            temp_expected = [max((x, eps)) for x in temp_expected]
            temp_result = np.random.dirichlet(temp_expected).tolist()
            # update stored transition probability
            self.p_emission[s] = {e: p for e, p in zip(self.emissions, temp_result)}

        # add emission probabilities from unseen states
        temp_expected = [self.alpha_emission * b for b in self.beta_emission.values()]
        temp_expected = [max((x, eps)) for x in temp_expected]
        temp_result = np.random.dirichlet(temp_expected).tolist()
        self.p_emission[None] = {e: p for e, p in zip(self.emissions, temp_result)}

    def print_fit_parameters(self):
        # TODO: print n_initial
        # create copies to avoid editing
        # n_initial = copy.deepcopy(self.n_initial)
        n_emission = copy.deepcopy(self.n_emission)
        n_transition = copy.deepcopy(self.n_transition)

        # make nested lists for clean printing
        emissions = [[str(s)] + [str(n_emission[s][e]) for e in self.emissions] for s in self.states]
        emissions.insert(0, ['S_i \ E_i'] + list(map(str, self.emissions)))
        transitions = [[str(s1)] + [str(n_transition[s1][s2]) for s2 in self.states] for s1 in self.states]
        transitions.insert(0, ['S_i \ S_j'] + list(map(lambda x: str(x), self.states)))

        # format tables
        te = DoubleTable(emissions, 'Emission counts')
        tt = DoubleTable(transitions, 'Transition counts')
        te.padding_left = 1
        te.padding_right = 1
        tt.padding_left = 1
        tt.padding_right = 1
        te.justify_columns[0] = 'right'
        tt.justify_columns[0] = 'right'

        # print tables
        print('\n')
        print(te.table)
        print('\n')
        print(tt.table)
        print('\n')

        #
        return None

    def print_probabilities(self):
        # TODO: print p_initial
        # create copies to avoid editing
        # p_initial = copy.deepcopy(self.p_initial)
        p_emission = copy.deepcopy(self.p_emission)
        p_transition = copy.deepcopy(self.p_transition)

        # convert to nested lists for clean printing
        p_emission = [[str(s)] + [str(round(p_emission[s][e], 3)) for e in self.emissions] for s in self.states]
        p_transition = [[str(s1)] + [str(round(p_transition[s1][s2], 3)) for s2 in self.states] for s1 in self.states]
        p_emission.insert(0, ['S_i \ E_j'] + [str(e) for e in self.emissions])
        p_transition.insert(0, ['S_i \ E_j'] + [str(s) for s in self.states])

        # format tables
        te = DoubleTable(p_emission, 'Emission probabilities')
        tt = DoubleTable(p_transition, 'Transition probabilities')
        te.padding_left = 1
        te.padding_right = 1
        tt.padding_left = 1
        tt.padding_right = 1
        te.justify_columns[0] = 'right'
        tt.justify_columns[0] = 'right'

        # print tables
        print('\n')
        print(te.table)
        print('\n')
        print(tt.table)
        print('\n')

        #
        return None

    def neglogp_sequences(self):
        return sum(chain.neglogp_sequence(self.p_initial, self.p_emission, self.p_transition) for chain in self.chains)

    def resample_chains(self, ncores=-1):
        # extract probabilities
        p_initial, p_emission, p_transition = self.p_initial, self.p_emission, self.p_transition

        # non-parallel execution:
        if ncores < 2:
            [chain.resample_latent_sequence(
                list(self.states)+[None], p_initial, p_emission, p_transition
            ) for chain in self.chains]

        # parallel execution
        else:
            # update auxiliary beam variables manually (fast; no need to parallelise)
            [chain.resample_auxiliary_beam_variables(p_initial, p_transition) for chain in self.chains]

            # create temporary function for mapping
            resample_partial = partial(
                Chain.resample_latent_sequence_static,
                states=list(self.states)+[None],
                p_initial=copy.deepcopy(p_initial),
                p_emission=copy.deepcopy(p_emission),
                p_transition=copy.deepcopy(p_transition))

            # parallel process resamples
            with multiprocessing.Pool(processes=ncores) as pool:
                results = pool.map(
                    resample_partial,
                    ((chain.emission_sequence, chain.auxiliary_beam_variables) for chain in self.chains))

            # assign returned latent sequences back to Chains (not done is static version)
            for i in range(self.c):
                self.chains[i].latent_sequence = results[i]

        # update chains using results
        # TODO: parameter check if we should be using alpha or gamma as parameter
        state_generator = dirichlet_process_generator(self.gamma, output_generator=lambda: self.generate_state())
        for chain in self.chains:
            chain.latent_sequence = [s if s is not None else next(state_generator) for s in chain.latent_sequence]

    def maximise_hyperparameters(self):
        pass

    def resample_hyperparameters(self):
        # use Metropolis Hastings resampling

        # get current and proposed hyperparameters
        hyperparameters_current = copy.deepcopy([
            self.alpha,
            self.gamma,
            self.alpha_emission,
            self.gamma_emission,
            self.kappa])
        hyperparameters_next = copy.deepcopy(hyperparameters_current)
        hyperparameters_proposed = [
            self.alpha_prior(),
            self.gamma_prior(),
            self.alpha_emission_prior(),
            self.gamma_emission_prior(),
            self.kappa_prior()]

        # iterate and accept each in order
        for param_index in range(len(hyperparameters_current)):
            # don't update kappa if not a sticky chain
            if not self.sticky and param_index == 4:
                continue

            # get current negative log likelihood
            neglogp_current = self.neglogp_sequences()

            # proposed new value for current parameter only
            hyperparameters_next[param_index] = hyperparameters_proposed[param_index]
            self.alpha, self.gamma, self.alpha_emission, self.gamma_emission, self.kappa = tuple(hyperparameters_next)

            # check new negative loglikelihood
            neglogp_proposed = self.neglogp_sequences()

            # find Metropolis Hasting acceptance probability
            p_accept = min(1, np.exp(- neglogp_proposed + neglogp_current))

            # choose whether to accept
            alpha_accepted = bool(np.random.binomial(n=1, p=p_accept))

            # if we do not accept, revert to the previous value
            if alpha_accepted:
                hyperparameters_current = copy.deepcopy(hyperparameters_next)
            else:
                hyperparameters_next = copy.deepcopy(hyperparameters_current)
                self.alpha, self.gamma, self.alpha_emission, self.gamma_emission, self.kappa = \
                    tuple(hyperparameters_next)

        return None

    def mcmc(self, n=1000, burn_in=500, save_every=10, ncores=-1):
        # store parameters
        state_count_hist = list()
        sequence_prob_hist = list()
        hyperparameter_hist = list()
        beta_emission_hist = list()
        beta_transition_hist = list()
        parameter_hist = list()

        # make sure auxiliary variables are in proper position before starting
        # TODO: this should be obsolete now
        [c.resample_auxiliary_beam_variables(self.p_initial, self.p_transition) for c in self.chains]

        # initialise other variables
        states_prev = copy.copy(self.states)

        for i in tqdm(range(n)):
            # complete one Gibbs iteration, working from higher level to lower level variables
            # hyperparameters need correct dimensional probabilities, so sample later
            self.resample_beta_transition(ncores=1)  # beta transition (inc. auxiliary variables)
            self.resample_beta_emission()  # beta emission
            self.resample_p_initial()  # probabilities
            self.resample_p_transition()
            self.resample_p_emission()
            self.resample_hyperparameters()  # hyperparameters
            self.resample_chains(ncores=ncores)  # latent sequence (including auxiliary beam variables)
            self.n_update()

            # check states and inform if number changes
            if len(states_prev) != self.k or self.states - states_prev != set():
                states_removed = states_prev - self.states
                states_added = self.states - states_prev
                tqdm.write(
                    'Iter {i}: state count from {k1} to {k2}, removed {s1}, introduced {s2}'.format(
                        # 'increased' if len(states_prev) < self.K else 'decreased',
                        i=i,
                        k1=len(states_prev),
                        k2=self.k,
                        s1=states_removed,
                        s2=states_added))
                states_prev = copy.copy(self.states)

            # store results
            if i >= burn_in and i % save_every == 0:
                # get parameters as nested lists (better import to R)
                p_initial = copy.deepcopy(self.p_initial)
                p_emission = copy.deepcopy(self.p_emission)
                p_transition = copy.deepcopy(self.p_transition)

                # save new data
                state_count_hist.append(self.k)
                sequence_prob_hist.append(self.neglogp_sequences())
                hyperparameter_hist.append((self.alpha,
                                            self.gamma,
                                            self.alpha_emission,
                                            self.gamma_emission,
                                            self.kappa))
                beta_emission_hist.append(self.beta_emission)
                beta_transition_hist.append(self.beta_transition)
                parameter_hist.append((p_initial, p_emission, p_transition))

        # return saved observations
        return (state_count_hist,
                sequence_prob_hist,
                hyperparameter_hist,
                beta_emission_hist,
                beta_transition_hist,
                parameter_hist)
