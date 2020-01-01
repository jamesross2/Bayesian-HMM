import typing

import scipy.stats

from .. import utils
from . import hyperparameter, stick_breaking_process, symbol, variable


class DirichletProcess(variable.Variable):
    def __init__(
        self,
        beta: stick_breaking_process.StickBreakingProcess,
        gamma: hyperparameter.Hyperparameter,
        kappa: typing.Optional[hyperparameter.Hyperparameter] = None,
    ) -> None:
        """The Dirichlet process gives the (infinite) transition probabilities of the hidden Markov model.

        This object is actually a family of Dirichlet processes, which share a common hierarchical prior (given by
        beta, another Dirichlet process, modelled as a stick breaking process). Since beta can grow in size (in number
        of states), the Dirichlet processes can grow in size also. This facilitates the infinite size of the
        hierarchical Dirichlet process.

        Args:
            beta: The stick breaking process describing the hierarchical priors for each Dirichlet process. States with
                large associated beta values are more likely to have large transition probabilities.
            gamma: The hyperparameter associated with each Dirichlet process.
            kappa: The hyperparameter associated with the stickiness of each Dirichlet process.

        """
        # init parent
        super(DirichletProcess, self).__init__()

        # store parents
        self.beta: stick_breaking_process.StickBreakingProcess = beta
        self.gamma: hyperparameter.Hyperparameter = gamma
        self.kappa: typing.Optional[hyperparameter.Hyperparameter] = kappa

        # fill with empty initial value
        self.value: typing.Dict[symbol.Symbol, typing.Dict[symbol.Symbol, float]] = {
            symbol.EmptySymbol(): {symbol.EmptySymbol(): 1}
        }

    @property
    def sticky(self) -> bool:
        """If True, then the DirichletProcess has a kappa variable and favours self-transitions.

        Returns:
            If True, then the Dirichlet Process is sticky.

        """
        return self.kappa is not None

    def posterior_parameters(
        self, states: typing.Set[symbol.Symbol], counts: typing.Dict[symbol.Symbol, typing.Dict[symbol.Symbol, int]]
    ) -> typing.Dict[symbol.Symbol, typing.Dict[symbol.Symbol, float]]:
        """The parameters of the posterior distribution.

        In order to resample the Dirichlet process from the posterior distribution, we use this method to get parameters
        for the posterior.

        Args:
            states: The states for which we wish to calculate parameters.
            counts: The number of transitions between states. Should include keys for every Symbol in `states`.

        Returns:
            A nested dictionary containing parameter values.

        """
        # kappa==0 implies non-sticky Dirichlet process
        kappa_value = self.kappa.value if self.sticky else 0.0

        # get parameters for posterior distribution
        # three components: count (posterior element), beta (prior), and kappa (sticky)
        params = {
            state0: {
                state1: counts.get(state0, dict()).get(state1, 0)
                + self.gamma.value * (1 - kappa_value) * self.beta.value.get(state1)
                + self.gamma.value * kappa_value * (state0 == state1)
                for state1 in states
            }
            for state0 in states
        }

        # parameters fully updated
        return params

    def log_likelihood(self) -> float:
        """The unconditional log likelihood of the Dirichlet process.

        This uses the prior distribution only, and ignores the transition counts.

        Returns:
            The log likelihood as a float (not necessarily negative).

        """
        # kappa==0 implies non-sticky Dirichlet process
        kappa_value = self.kappa.value if self.sticky else 0.0
        states = tuple(self.value.keys())

        # extract parameters for prior distribution
        parameters = {
            state0: tuple(
                self.gamma.value * (1 - kappa_value) * self.beta.value.get(state1)
                + self.gamma.value * kappa_value * (state0 == state1)
                for state1 in states
            )
            for state0 in states
        }

        # iterate within a loop to ensure that unordered dicts match states properly
        log_likelihoods = {}
        for state in states:
            # TODO: consider if this is the right place to shrink; or should be done elsewhere.
            values_shrunk = utils.shrink_probabilities((self.value[state]))
            values = tuple(values_shrunk[s] for s in states)
            log_likelihoods[state] = scipy.stats.dirichlet.logpdf(values, parameters[state])

        return sum(log_likelihoods.values())

    def resample(
        self, states: typing.Set[symbol.Symbol], counts: typing.Dict[symbol.Symbol, typing.Dict[symbol.Symbol, int]]
    ) -> typing.Dict[symbol.Symbol, typing.Dict[symbol.Symbol, float]]:
        """Repopulate the Dirichlet processes with new samples.

        Essentially draws a new transition matrix from the current stick breaking process and observed transition
        counts.

        Args:
            states: States to include in the resampling step.
            counts: The number of state transitions of the latent series.

        Returns:
            The resampled value.
        """
        # get parameters for posterior distribution
        parameters = self.posterior_parameters(states=states, counts=counts)

        # resample each state's transition matrix
        value = {
            state: dict(
                zip(parameters[state].keys(), scipy.stats.dirichlet.rvs(alpha=list(parameters[state].values()))[0])
            )
            for state in states
        }

        # update object
        self.value = value
        return self.value

    def add_state(self, state: symbol.Symbol) -> None:
        """Add a state to the family of Dirichlet processes, without resampling all states.

        Args:
            state: The new state to include.

        """
        # First task will be to break out new symbol from existing Dirichlet processes
        for state_from in self.value.keys():
            temp_beta = scipy.stats.beta.rvs(1, self.gamma.value)
            temp_value = self.value.get(state_from).get(symbol.EmptySymbol())
            self.value[state_from][state] = temp_beta * temp_value
            self.value[state_from][symbol.EmptySymbol()] = (1.0 - temp_beta) * temp_value

        # Second task will be to create a new Dirichlet process for the new symbol
        # Posterior parameters have three components: count (posterior element), beta (prior), and kappa (sticky)
        kappa_value = self.kappa.value if self.sticky else 0.0
        states = self.beta.value.keys()
        parameters = {
            state1: self.gamma.value * (1 - kappa_value) * self.beta.value.get(state1)
            + self.gamma.value * kappa_value * (state == state1)
            for state1 in states
        }
        value = dict(zip(parameters.keys(), scipy.stats.dirichlet.rvs(alpha=list(parameters.values()))[0]))
        self.value[state] = value

    def remove_state(self, state: symbol.Symbol) -> None:
        raise NotImplementedError("TODO: add remove_state functionality for DirichletProcess.")
