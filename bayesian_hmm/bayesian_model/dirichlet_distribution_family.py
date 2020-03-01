import collections
import copy
import typing

import scipy.stats

from .. import utils
from . import hyperparameter, states, variable


class DirichletDistributionFamily(variable.Variable):
    def __init__(
        self,
        beta: typing.Union[hyperparameter.Hyperparameter, typing.Dict[states.State, hyperparameter.Hyperparameter]],
    ) -> None:
        """The Dirichlet process gives the (infinite) transition probabilities of the hidden Markov model.

        This object is actually a family of Dirichlet processes, which share a common hierarchical prior (given by
        beta, another Dirichlet process, modelled as a stick breaking process). Since beta can grow in size (in number
        of states), the Dirichlet processes can grow in size also. This facilitates the infinite size of the
        hierarchical Dirichlet process.

        Args:
            beta: The prior for each Dirichlet distribution. This should be either a vector of hyperparameters giving
                the prior for the finite states (or a single Hyperparameter, in which case every category will have the
                same prior). States with large associated beta values are more likely to have large transition
                probabilities.

        Raises:
            ValueError: if the given beta doesn't match any expected input type.

        """
        # init parent
        super(DirichletDistributionFamily, self).__init__()

        #
        beta_factory: typing.Mapping[states.State, hyperparameter.Hyperparameter]
        if isinstance(beta, hyperparameter.Hyperparameter):
            beta_factory = collections.defaultdict(lambda: copy.deepcopy(beta))  # type: ignore
        elif isinstance(beta, dict):
            beta_factory = beta
        else:
            raise ValueError("Unrecognised beta type {}, use a hyperparameter or dict.".format(type(beta)))

        # store parents
        self._beta_factory: typing.Mapping[states.State, hyperparameter.Hyperparameter] = beta_factory

        # fill with empty initial value
        self.value: typing.Dict[states.State, typing.Dict[states.State, float]] = dict()

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
            counts: The number of transitions between states. Should include keys for every State in `states`.

        Returns:
            A nested dictionary containing parameter values.

        """
        # fill in default values
        states_from = tuple(counts.keys())
        states_to: typing.Sequence[states.State]
        if len(states_from) == 0:
            states_to = tuple()
        else:
            states_to = tuple(counts[states_from[0]].keys())

        # get parameters for posterior distribution
        parameters = {
            state_from: {
                state_to: counts.get(state_from, dict()).get(state_to, 0) + self._beta_factory[state_to].value
                for state_to in states_to
            }
            for state_from in states_from
        }

        # parameters fully updated
        return parameters

    def log_likelihood(self) -> float:
        """The unconditional log likelihood of the Dirichlet process.

        This uses the prior distribution only, and ignores the transition counts.

        Returns:
            The log likelihood as a float (not necessarily negative).

        """
        # extract parameters for prior distribution
        parameters = tuple(self._beta_factory[state_to].value for state_to in self.states_outer)

        # iterate within a loop to ensure that unordered dicts match states properly
        log_likelihoods = {}
        for state_from in self.states_inner:
            values: typing.Sequence[float] = tuple(self.value[state_from][state_to] for state_to in self.states_outer)
            values = tuple(utils.shrink_probabilities(values))
            log_likelihoods[state_from] = scipy.stats.dirichlet.logpdf(values, parameters)

        return sum(log_likelihoods.values())

    def resample(
        self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]
    ) -> typing.Dict[states.State, typing.Dict[states.State, float]]:
        """Repopulate the Dirichlet distribution.

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
            inner: If True (the default), then the new state will be added to the inner keys for the Dirichlet process.
            outer: If True (the default), then the new state will be added to the outer keys for the Dirichlet process.

        """
        if outer:
            for state_from in self.states_inner:
                if len(self.value[state_from]) > 0:
                    self.value[state_from][state] = 0.0
                else:
                    self.value[state_from] = {state: 1.0}

        if inner:
            if len(self.states_outer) > 0:
                parameters = {state_to: self._beta_factory[state].value for state_to in self.states_outer}
                value = dict(zip(parameters.keys(), scipy.stats.dirichlet.rvs(alpha=list(parameters.values()))[0]))
            else:
                value = dict()
            self.value[state] = value

    def remove_state(self, state: states.State, inner: bool = True, outer: bool = True) -> None:
        """Drops a state from the Dirichlet process family; both its own transitions and the transitions into it.

        Args:
            state: The state to remove.
            inner: If True (the default), then the new state will be removed from the inner keys for the Dirichlet
                family.
            outer: If True (the default), then the new state will be removed from the outer keys for the Dirichlet
                family.

        """
        if inner:
            del self.value[state]

        if outer:
            for state_from in self.value.keys():
                del self.value[state_from][state]
