import typing

import scipy.stats

from .. import utils
from . import hyperparameter, states, stick_breaking_process, variable


class DirichletProcessFamily(variable.Variable):
    def __init__(
        self,
        beta: stick_breaking_process.StickBreakingProcess,
        gamma: hyperparameter.Hyperparameter,
        kappa: hyperparameter.Hyperparameter,
    ) -> None:
        """The Dirichlet process gives the (infinite) transition probabilities of the hidden Markov model.

        This object is actually a family of Dirichlet processes, which share a common hierarchical prior (given by
        beta, another Dirichlet process, modelled as a stick breaking process). Since beta can grow in size (in number
        of states), the Dirichlet processes can grow in size also. This facilitates the infinite size of the
        hierarchical Dirichlet process.

        Args:
            beta: The prior for each Dirichlet distribution. This should be a StickBreakingProcess, which gives a
                prior over the infinite states of the model. The stick breaking process describing the hierarchical
                priors for each Dirichlet process. States with large associated beta values are more likely to have
                large transition probabilities.
            gamma: The hyperparameter associated with each Dirichlet process.
            kappa: The hyperparameter associated with the stickiness of each Dirichlet process.

        """
        # init parent
        super(DirichletProcessFamily, self).__init__()

        # store parents
        self.beta: stick_breaking_process.StickBreakingProcess = beta
        self.gamma: hyperparameter.Hyperparameter = gamma
        self.kappa: hyperparameter.Hyperparameter = kappa

        # fill with empty initial value
        self.value: typing.Dict[states.State, typing.Dict[states.State, float]]
        self.value = {states.AggregateState(): {states.AggregateState(): 1}}

    @property
    def sticky(self) -> bool:
        """If True, then the DirichletProcess has a kappa variable and favours self-transitions.

        Returns:
            If True, then the Dirichlet Process is sticky.

        """
        return not isinstance(self.kappa, hyperparameter.Dummy)

    @property
    def states_inner(self) -> typing.Set[states.State]:
        """The variables parameterising the Dirichlet process realisations.

        Returns:
            The set of latent states currently in the family. This will include any special states if they have been
                given in the model. Latent states tie into the latent variables of the hidden Markov model, and not
                of the emission variables (see ``states_outer`` for these).

        """
        return set(self.value.keys())

    @property
    def states_outer(self) -> typing.Set[states.State]:
        """The variables parameterising the outcome distributions.

        Returns:
            The set of emission states currently in the family. This will include any special states if they have been
                given in the model. Emission states tie into the observed emissions of each chain, and not of teh latent
                states in a chain (see ``states_inner`` for these).

        """
        if len(self.value) == 0:
            return set()

        temp_key = list(self.value.keys())[0]
        return set(self.value[temp_key].keys())

    def posterior_parameters(
        self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]
    ) -> typing.Dict[states.State, typing.Dict[states.State, float]]:
        """The parameters of the posterior distribution.

        In order to resample the Dirichlet process from the posterior distribution, we use this method to get parameters
        for the posterior.

        Args:
            counts: The number of transitions between states. Should include keys for every Symbol in `states`.

        Returns:
            A nested dictionary containing parameter values.

        """
        # fill in default values
        states_from = set(counts.keys())
        states_to: typing.Set
        if len(states_from) == 0:
            states_to = set()
        else:
            states_to = set(counts[list(states_from)[0]].keys())

        # add aggregate states
        states_from.add(states.AggregateState())
        states_to.add(states.AggregateState())

        # get parameters for posterior distribution
        # three components: count (posterior element), beta (prior), and kappa (sticky)
        parameters = {
            state0: {
                state1: counts.get(state0, dict()).get(state1, 0)
                + self.gamma.value * (1 - self.kappa.value) * self.beta.value[state1]
                + self.gamma.value * self.kappa.value * (state0 == state1)
                for state1 in states_to
            }
            for state0 in states_from
        }

        # parameters fully updated
        return parameters

    def log_likelihood(self) -> float:
        """The unconditional log likelihood of the Dirichlet process.

        This uses the prior distribution only, and ignores the transition counts.

        Returns:
            The log likelihood as a float (not necessarily negative).

        """
        states_from = self.states_inner
        states_to = self.states_outer

        # extract parameters for prior distribution
        parameters = {
            state0: tuple(
                self.gamma.value * (1 - self.kappa.value) * self.beta.value[state1]
                + self.gamma.value * self.kappa.value * (state0 == state1)
                for state1 in states_to
            )
            for state0 in states_from
        }

        # iterate within a loop to ensure that unordered dicts match states properly
        log_likelihoods = {}
        for state_from in states_from:
            values: typing.Sequence[float] = tuple(self.value[state_from][state_to] for state_to in states_to)
            values = utils.shrink_probabilities(values)
            log_likelihoods[state_from] = scipy.stats.dirichlet.logpdf(
                values, utils.shrink_probabilities(parameters[state_from])
            )

        return sum(log_likelihoods.values())

    def resample(
        self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]
    ) -> typing.Dict[states.State, typing.Dict[states.State, float]]:
        """Repopulate the Dirichlet processes with new samples.

        Essentially draws a new transition matrix from the current stick breaking process and observed transition
        counts.

        Args:
            counts: The number of state transitions of the latent series.

        Returns:
            The resampled value.

        """
        # get parameters for posterior distribution
        parameters = self.posterior_parameters(counts=counts)

        # resample each state's transition matrix
        value = {
            state_from: dict(
                zip(
                    parameters[state_from].keys(),
                    scipy.stats.dirichlet.rvs(alpha=list(parameters[state_from].values()))[0],
                )
            )
            for state_from in counts.keys()
        }

        # update object
        self.value = value
        return self.value

    def add_state(self, state: states.State, inner: bool = True, outer: bool = True) -> None:
        """Add a state to the family of Dirichlet processes, without resampling all states.

        We separate the inner and outer keys by using inner and outer flags. This lets us
        use the same Dirichlet process to model multinomial distributions with different sets
        of latent states and categories.

        Args:
            state: The new state to include.
            inner: If True (the default), then the new symbol will be added to the inner keys for the Dirichlet process.
            outer: If True (the default), then the new symbol will be added to the outer keys for the Dirichlet process.

        """
        if outer:
            for state_from in self.states_inner:
                temp_beta = scipy.stats.beta.rvs(1, self.gamma.value)
                temp_value = self.value[state_from][states.AggregateState()]
                self.value[state_from][state] = temp_beta * temp_value
                self.value[state_from][states.AggregateState()] = (1.0 - temp_beta) * temp_value

        if inner:
            states_to = self.states_outer
            parameters = {
                state_to: self.gamma.value * (1 - self.kappa.value) * self.beta.value[state_to]
                + self.gamma.value * self.kappa.value * (state == state_to)
                for state_to in states_to.union({state})
            }
            parameters[states.AggregateState()] = self.gamma.value
            parameters = utils.shrink_probabilities(parameters)
            value = dict(zip(parameters.keys(), scipy.stats.dirichlet.rvs(alpha=list(parameters.values()))[0]))
            self.value[state] = value

    def remove_state(self, state: states.State, inner: bool = True, outer: bool = True) -> None:
        """Drops a state from the Dirichlet process family; both its own transitions and the transitions into it.

        Args:
            state: The state to remove.
            inner: If True (the default), then the new symbol will be added to the inner keys for the Dirichlet process.
            outer: If True (the default), then the new symbol will be added to the outer keys for the Dirichlet process.

        """
        if inner:
            del self.value[state]

        if outer:
            for state_from in self.value.keys():
                self.value[state_from][states.AggregateState()] += self.value[state_from][state]
                del self.value[state_from][state]
