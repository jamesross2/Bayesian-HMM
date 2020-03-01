"""A simple implementation for non-parametric Hierarchical Dirichlet process hidden Markov models.

Hierarchical Dirichlet Process Hidden Markov Model (HDPHMM).
The HDPHMM object collects a number of observed emission sequences, and estimates
latent states at every time point, along with a probability structure that ties latent
states to emissions. This structure involves

  + A starting probability, which dictates the probability that the first state
      in a latent sequence is equal to a given symbol. This has a hierarchical Dirichlet
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

import copy
import functools
import multiprocessing
import string
import typing

import numpy
import terminaltables
import tqdm

from . import bayesian_model, utils
from .bayesian_model import hyperparameter
from .chain import Chain, resample_latent_sequence

# Shorthand for numeric types.
Numeric = typing.Union[int, float]

# Oft-used dictionary initializations with shorthands.
InitDict = typing.Dict[bayesian_model.State, Numeric]
NestedInitDict = typing.Dict[bayesian_model.State, InitDict]


# TODO: check docstring
class HDPHMM(object):
    """
    The Hierarchical Dirichlet Process Hidden Markov Model object. In fact, this is a
    sticky-HDPHMM, since we allow a biased self-transition probability.
    """

    def __init__(
        self,
        emission_sequences: typing.Iterable[typing.Sequence[bayesian_model.State]],
        emissions: typing.Optional[typing.Set[bayesian_model.State]] = None,
        sticky: bool = True,
        alpha: hyperparameter.Hyperparameter = hyperparameter.Gamma(shape=2, scale=2),
        gamma: hyperparameter.Hyperparameter = hyperparameter.Gamma(shape=3, scale=3),
        kappa: bayesian_model.Hyperparameter = hyperparameter.Beta(shape=1, scale=1),
        beta_emission: bayesian_model.Hyperparameter = hyperparameter.Gamma(shape=2, scale=2),
    ) -> None:
        """A fully non-parametric Bayesian hierarchical Dirichlet process hidden Markov model.

        Create a Hierarchical Dirichlet Process Hidden Markov Model object, which can
        (optionally) be sticky. The emission sequences must be provided, although all
        other parameters are initialised with reasonable default values. It is
        recommended to specify the `sticky` parameter, depending on whether you believe
        the HMM to have a high probability of self-transition.

        The 'non-parametric' description means that the number of hidden states (analogous to
        the number of clusters in a clustering algorithm) does not need to be set. Instead,
        the model will automatically explore (using a Gibbs sampling approach) different
        numbers of latent states. This is analogous to the non-parametric clustering
        algorithms, which we recommend for the reader trying to introduce themselves to
        non-parametric Bayesian methods.

        Given that non-parametric applies to the number of latent states, there are still
        some parameters which govern the dynamics of the model. Parameters are initialised
        with reasonable default values, but can be easily replaced. A description of each
        parameter's role is below.
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
                likely to explore states with high self-transition probability.

        Args:
            emission_sequences: An iterable containing the observed emission sequences.
                emission sequences can be different lengths, or zero length.
            emissions: If not all emissions are guaranteed to be
                observed in the data, this can be used to force non-zero emission probabilities
                for unobserved emissions.
            sticky: flag to indicate whether the HDPHMM is sticky or not.
                Sticky HDPHMMs have an additional value (kappa) added to the probability of self
                transition. It is recommended to set this depending on the knowledge of the
                problem at hand.
            alpha: A Hyperparameter governing alpha.
            gamma: A hyperparameter governing gamma.
            kappa: A hyperparameter governing kappa. Ignored if sticky is False.
            beta_emission: A hyperparameter governing the beta of the emission distribution.

        Raises:
            ValueError: for invalid combinations of inputs.

        """
        # store chains
        self.chains = [Chain(sequence) for sequence in emission_sequences]

        # sticky flag
        if type(sticky) is not bool:
            raise ValueError("`sticky` must be type bool")
        elif sticky and isinstance(kappa, hyperparameter.Dummy):
            raise ValueError("Hyperparameter kappa must be non-dummy for a sticky process.")
        elif not sticky:
            # replace kappa with None if the process is not sticky
            kappa = hyperparameter.Dummy(0.0)

        # store raw hyperparameter values for convenience
        self.sticky = sticky
        self.alpha: hyperparameter.Hyperparameter = alpha
        self.gamma: hyperparameter.Hyperparameter = gamma
        self.kappa: hyperparameter.Hyperparameter = kappa
        self.beta_emission: hyperparameter.Hyperparameter = beta_emission

        # create a hierarchical Dirichlet process model to store the Bayesian transition dynamics
        self.transition_model: bayesian_model.HierarchicalDirichletProcess
        self.emission_model: bayesian_model.HierarchicalDirichletDistribution
        self.transition_model = bayesian_model.HierarchicalDirichletProcess(
            sticky=self.sticky, alpha=self.alpha, gamma=self.gamma, kappa=self.kappa
        )
        self.emission_model = bayesian_model.HierarchicalDirichletDistribution(beta=self.beta_emission)

        # use internal properties to store aggregate statistics (used to update Bayesian variables efficiently)
        self.emission_counts: typing.Dict[bayesian_model.State, typing.Dict[bayesian_model.State, int]] = {}
        self.transition_counts: typing.Dict[bayesian_model.State, typing.Dict[bayesian_model.State, int]] = {}

        # states & emissions
        if emissions is None:
            emissions = set(emission for chain in self.chains for emission in chain.emission_sequence)
        elif not isinstance(emissions, set):
            raise ValueError("emissions must be a set")
        assert isinstance(emissions, set)
        self.emissions: typing.Set[bayesian_model.State] = emissions
        self.states: typing.Set[bayesian_model.State] = set()

        # generate non-repeating character labels for latent states
        self._label_generator = utils.label_generator(string.ascii_lowercase)

        # keep flag to track initialisation
        self._initialised = False

    @property
    def initialised(self) -> bool:
        """Test whether a HDPHMM is initialised.

        Returns:
            True if the chain has been initialised.

        """
        return self._initialised

    @initialised.setter
    def initialised(self, value: typing.Any) -> None:
        if value:
            raise AssertionError("HDPHMM must be initialised through initialise method")
        elif not value:
            self._initialised = False
        else:
            raise ValueError("initialised flag must be Boolean")

    @property
    def c(self) -> int:
        """Number of chains in the HMM.

        Returns:
            The number of chains in the model currently.

        """
        return len(self.chains)

    @property
    def k(self) -> int:
        """Number of latent states in the HMM currently.

        Returns:
            The number of states, excluding the AggregateState and StartingState states.
        """
        return len(self.states)

    @property
    def n(self) -> int:
        """Number of unique emissions.

        If `emissions` was specified when the HDPHMM was
        created, then this counts the number of elements in `emissions`. Otherwise,
        counts the number of observed emissions across all emission sequences.

        Returns:
            An emission count.

        """
        return len(self.emissions)

    def tabulate(self) -> numpy.array:
        """Create a table containing the state label of every emission and chain.

        Convert the latent and emission sequences for all chains into a single numpy
        array. Array contains an index which matches a Chain's index in
        HDPHMM.chains, the current latent state, and the emission for all chains at
        all times.

        Returns:
            A numpy array with dimension (l, 3), where l is the length of the Chain

        """
        hmm_array = numpy.concatenate(
            tuple(
                numpy.concatenate((numpy.array([[n] * self.chains[n].T]).T, self.chains[n].tabulate()), axis=1)
                for n in range(self.c)
            ),
            axis=0,
        )
        return hmm_array

    def __repr__(self) -> str:
        return "<bayesian_hmm.HDPHMM, size {C}>".format(C=self.c)

    def __str__(self, print_len: int = 15) -> str:
        fs = "bayesian_hmm.HDPHMM," + " ({C} chains, {K} states, {N} emissions, {Ob} observations)"
        return fs.format(C=self.c, K=self.k, N=self.n, Ob=sum(c.T for c in self.chains))

    def state_generator(self) -> typing.Generator[bayesian_model.State, None, None]:
        """Create a new state for the HDPHMM, and update all parameters accordingly.

        Yields:
            The label of the new state.

        """
        while True:
            # make a state with a new label
            label = next(self._label_generator)
            state = bayesian_model.State(label)
            self.states.add(state)

            # add the state to the hierarchical process
            self.emission_model.add_state(state)
            self.transition_model.add_state(state)

            # new states have no observations, so are not observed yet
            self.transition_counts[state] = {s: 0 for s in self.states}
            for state_other in self.states:
                self.transition_counts[state_other][state] = 0
            self.emission_counts[state] = {e: 0 for e in self.emissions}

            # finished
            yield state

    def initialise(self, k: int = 20) -> None:
        """Randomly assign latent states to observations.

        Initialisation involves:
            + Initialising the hierarchical Dirichlet process.
            + Initialising all Chains (see Chain.initialise for further info).
            + Updating all counts

        Args:
            k: The number of symbols to sample from for latent states.

        """
        # create as many states as needed
        labels = [next(self._label_generator) for _ in range(k)]
        states = {bayesian_model.State(label) for label in labels}
        self.states.update(states)

        # initialise chains
        for c in self.chains:
            c.initialise(self.states)

        # initialise transition and emission models with augmented states
        for state in states:
            self.transition_model.add_state(state)
            self.emission_model.add_state(state, inner=True, outer=False)

        # add emissions to emission model
        for emission in self.emissions:
            self.emission_model.add_state(emission, inner=False, outer=True)

        # update state counts
        self.update_counts()

        # resample the states in the models for non-degenerate starting conditions
        self.emission_model.resample(counts=self.emission_counts)
        self.transition_model.resample(counts=self.transition_counts)

        # set initialised flag
        self._initialised = True

    def update_states(self):
        """Remove defunct states from transition and emission dynamics."""
        states_prev = self.states
        states_next = set(sorted(set(emission for chain in self.chains for emission in chain.latent_sequence)))
        states_removed = (states_prev - states_next) - {bayesian_model.AggregateState(), bayesian_model.StartingState()}

        # merge old probabilities into None
        for state in states_removed:
            # remove state from internal models
            self.transition_model.remove_state(state)
            self.emission_model.remove_state(state, inner=True, outer=False)

            # remove state from attributes
            del self.transition_counts[state]
            del self.emission_counts[state]

            for state_from in states_next:
                del self.transition_counts[state_from][state]

        # update internal state tracking
        self.states = states_next

    def update_counts(self):
        """Update counts required for resampling probabilities.

        These counts are used to sample from the posterior distribution for probabilities.
        This function should be called after any latent state is changed, including after
        resampling.

        Raises:
            AssertionError: If the model is not yet initialised.

        """
        # check that all chains are initialised
        if any(not chain.initialised_flag for chain in self.chains):
            raise AssertionError("Chains must be initialised before calculating fit parameters")

        # transition count for non-oracle transitions
        emission_counts = {s: {e: 0 for e in self.emissions} for s in self.states}
        transition_counts = {
            s1: {s2: 0 for s2 in self.states} for s1 in self.states.union({bayesian_model.StartingState()})
        }

        # increment all relevant hyperparameters while looping over sequence
        for chain in self.chains:
            # increment transition from special starting state
            if chain.T > 0:
                transition_counts[bayesian_model.StartingState()][chain.latent_sequence[0]] += 1

            # increment all transitions and emissions within chain
            for t in range(chain.T - 1):
                emission_counts[chain.latent_sequence[t]][chain.emission_sequence[t]] += 1
                transition_counts[chain.latent_sequence[t]][chain.latent_sequence[t + 1]] += 1

            # increment emissions only for final state
            emission_counts[chain.latent_sequence[chain.T - 1]][chain.emission_sequence[chain.T - 1]] += 1

        # store recalculated fit hyperparameters
        self.emission_counts = emission_counts
        self.transition_counts = transition_counts

    def print_probabilities(self, digits: int = 4) -> typing.Tuple[str, str]:
        """Create an ascii-printable version of the transition and emission parameters.

        Args:
            digits: decimal places to print

        Returns:
            emission parameters, transition parameters: two tables, each containing a
                table parameters.
        """
        # make nested lists for clean printing
        emissions = [
            [str(s)] + [str(round(self.emission_model.pi.value[s][e], digits)) for e in self.emissions]
            for s in self.states
        ]
        emissions.insert(0, ["S_i \\ E_i"] + list(map(str, self.emissions)))
        transitions = [
            [str(s1)] + [str(round(self.transition_model.pi.value[s1][s2], digits)) for s2 in self.states]
            for s1 in self.states.union({bayesian_model.StartingState()})
        ]
        transitions.insert(
            0, ["S_i \\ S_j"] + list(map(lambda x: str(x), self.states.union({bayesian_model.StartingState()})))
        )

        # format tables
        te = terminaltables.DoubleTable(emissions, "Emission probabilities")
        tt = terminaltables.DoubleTable(transitions, "Transition probabilities")
        te.padding_left = 1
        te.padding_right = 1
        tt.padding_left = 1
        tt.padding_right = 1
        te.justify_columns[0] = "right"
        tt.justify_columns[0] = "right"

        #
        return te.table, tt.table

    def chain_log_likelihoods(self) -> typing.List[float]:
        """Calculate the negative log likelihood of every chain in the model.

        This log likelihood depends on the current latent states. This is calculated based on
        the observed emission sequences only, and not on the probabilities of the hyperparameters.

        Returns:
            A list of the log likelihood for each Chain.

        """
        chain_log_likelihoods = [
            chain.log_likelihood(self.emission_model.pi.value, self.transition_model.pi.value) for chain in self.chains
        ]
        return chain_log_likelihoods

    def log_likelihood(self) -> float:
        """The full joint likelihood of the model and all observed data.

        Returns:
            The total log likelihood of the model, including the Hierarchical Dirichlet transition probabilities, the
                Dirichlet emission probabilities, and the latent state transition and emission probabilities.

        """
        log_likelihoods = (
            self.transition_model.log_likelihood(),
            self.emission_model.log_likelihood(),
            sum(self.chain_log_likelihoods()),
        )
        return sum(log_likelihoods)

    def resample_chains(self, ncores=1):
        """Resample the latent states in all chains.

        This uses Beam sampling to improve the resampling time.

        Args:
            ncores: The number of threads to use in multithreading. If less than 1, no parallelisation is used.

        """
        # extract probabilities
        p_emission, p_transition = (self.emission_model.pi.value, self.transition_model.pi.value)

        # create temporary function for mapping
        resample_partial = functools.partial(
            resample_latent_sequence,
            states=self.states,
            emission_probabilities=copy.deepcopy(p_emission),
            transition_probabilities=copy.deepcopy(p_transition),
        )

        # parallel process resamples
        pool = multiprocessing.Pool(processes=ncores)
        new_latent_sequences = pool.map(
            resample_partial, ((chain.emission_sequence, chain.latent_sequence) for chain in self.chains)
        )
        pool.close()

        # assign returned latent sequences back to Chains
        for i in range(self.c):
            self.chains[i].latent_sequence = new_latent_sequences[i]

        # update chains using results
        state_generator = utils.dirichlet_process_generator(
            alpha=self.transition_model.alpha.value, output_generator=self.state_generator()
        )
        for chain in self.chains:
            chain.latent_sequence = [s if s.value is not None else next(state_generator) for s in chain.latent_sequence]

        # update counts
        self.update_states()
        self.update_counts()

    def maximise_hyperparameters(self):
        """Choose the MAP (maximum a posteriori) value for the hyperparameters.

        This is not yet implemented.

        Raises:
            NotImplementedError: Raise an issue on the project GitHub if you would like this.

        """
        raise NotImplementedError("This has not yet been written!" + " Ping the author if you want it to happen.")

    def mcmc(
        self, n: int = 1000, burn_in: int = 500, save_every: int = 10, ncores: int = 1, verbose=True
    ) -> typing.Dict[str, typing.List[typing.Any]]:
        """Iterate resampling of the entire model to estimate parameters and model fit.

        Use Markov chain Monte Carlo to estimate the starting, transition, and emission
        parameters of the HDPHMM, as well as the number of latent states.

        Args:
            n: The number of iterations to complete.
            burn_in: The number of iterations to complete before savings results.
            save_every: only iterations which are a multiple of `save_every`
                will have their results appended to the results.
            ncores: number of cores to use in multithreaded latent state resampling.
            verbose: Flag to indicate whether iteration-level statistics should be printed.

        Returns:
            A dict containing results from every saved iteration. Includes:
                + the number of states of the HDPHMM
                + the negative log likelihood of the entire model
                + the negative log likelihood of the chains only
                + the hyperparameters of the HDPHMM
                + the emission beta values
                + the transition beta values
                + all probability dictionary objects

        """
        # store hyperparameters in a single dict
        results: typing.Dict[str, typing.List[typing.Any]] = {
            "state_count": list(),
            "loglikelihood": list(),
            "chain_loglikelihood": list(),
            "hyperparameters": list(),
            "beta_emission": list(),
            "beta_transition": list(),
            "emission_probabilities": list(),
            "transition_probabilities": list(),
        }

        for i in tqdm.tqdm(range(n)):
            # update statistics
            states_prev = copy.copy(self.states)

            # work down hierarchy when resampling
            self.resample_chains(ncores=ncores)
            self.emission_model.resample(counts=self.emission_counts)
            self.transition_model.resample(counts=self.transition_counts)

            # update computation-heavy statistics
            likelihood_curr = self.log_likelihood()

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
                # save new data
                results["state_count"].append(self.k)
                results["loglikelihood"].append(likelihood_curr)
                results["chain_loglikelihood"].append(sum(self.chain_log_likelihoods()))
                results["hyperparameters"].append(
                    (
                        self.emission_model.beta.value,
                        self.transition_model.alpha.value,
                        self.transition_model.gamma.value,
                        self.transition_model.kappa.value if self.sticky else None,
                    )
                )
                results["beta_emission"].append(copy.deepcopy(self.emission_model.beta.value))
                results["beta_transition"].append(copy.deepcopy(self.transition_model.beta.value))
                results["emission_probabilities"].append(copy.deepcopy(self.emission_model.pi.value))
                results["transition_probabilities"].append(copy.deepcopy(self.transition_model.pi.value))

        # return saved observations
        return results
