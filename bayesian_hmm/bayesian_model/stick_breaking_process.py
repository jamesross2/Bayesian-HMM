"""A stick breaking process implementation of the Dirichlet process."""

import typing

import numpy
import scipy.special
import scipy.stats

from . import auxiliary_variable, hyperparameter, states, variable


class StickBreakingProcess(variable.Variable):
    def __init__(self, alpha: hyperparameter.Hyperparameter) -> None:
        """The stick breaking process parametrises the prior for the transition probabilities.

        It is another Dirichlet process, but modelled by the interpretable stick-breaking process. It contains a
        partition of the unit interval into infinitely many intervals, with each interval's size given by a beta
        random variable multiplied by the remaining length (beta itself has distribution ``Beta(1, alpha)``). Of
        course, we do not capture the infinite number of states: instead, we aggregate the tail into a single aggregate
        state, which can be decomposed when required.

        Args:
            alpha: The hyperparameter governing the distribution of the stick breaking process.

        """
        # parent information is the alpha hyperparameter and auxiliary hyperparameters
        super(StickBreakingProcess, self).__init__()
        self.alpha: hyperparameter.Hyperparameter = alpha

        # fill with empty initial value
        self.value: typing.Dict[states.State, float] = {states.AggregateState(): 1}

    def log_likelihood(self) -> float:
        """The likelihood of the stick breaking process is the product of likelihoods of each component length.

        Each length is given by a beta variable (see the help for the object), so the log likelihood is simply a sum
        of beta variable likelihoods.

        Returns:
            The log likelihood (under the prior distribution) of the stick breaking process' current value.

        """
        # likelihood given as product of likelihoods of corresponding beta variables
        values = list(self.value[s] for s in self.value.keys() if s != states.AggregateState())
        betas = [val / (1 + val - cumval) for val, cumval in zip(values, numpy.cumsum(values))]
        log_likelihoods = [scipy.stats.beta.logpdf(x=b, a=1, b=self.alpha.value) for b in betas]
        return sum(log_likelihoods)

    def resample(self, auxiliary: "auxiliary_variable.AuxiliaryVariable") -> typing.Dict["states.State", float]:
        """Draw another realisation of the stick breaking process, according to the current conditional distribution.

        The conditional distribution is parametrised completely by the auxiliary variables, so these will govern the
        resampling probabilities.

        Args:
            auxiliary: The auxiliary variables, already resampling, which parametrise the conditional distribution for
                the stick breaking process.

        Returns:
            The new value of beta.

        """
        # extract aggregate auxiliary variables to get parameters of conditional distribution for beta
        parameters: typing.Dict[states.State, typing.Union[int, float]] = dict(auxiliary.value_aggregated())
        parameters[states.AggregateState()] = self.alpha.value

        # resample using parameters of conditional distribution
        values = scipy.stats.dirichlet.rvs(alpha=list(parameters.values()), size=1)[0]

        # associate values back to states
        self.value = dict(zip(parameters.keys(), values))
        return self.value

    def add_state(self, state: states.State) -> None:
        """Separates another state from the aggregate ``EmptySymbol`` to be explicitly included.

        Args:
            state: The state to be added to the stick breaking process.

        Raises:
            ValueError: If the state is already included in the stick breaking process.

        """
        if state in self.value.keys():
            raise ValueError(f"State {state} already included in the stick breaking process {self}.")

        temp_beta = scipy.stats.beta.rvs(a=1, b=self.alpha.value)
        self.value[state] = temp_beta * self.value.get(states.AggregateState())
        self.value[states.AggregateState()] = (1.0 - temp_beta) * self.value.get(states.AggregateState())

    def remove_state(self, state: states.State) -> None:
        """Drops a state from the process and adds the parameter back into the aggregate state.

        Args:
            state: The state to be removed from the process.

        """
        self.value[states.AggregateState()] += self.value[state]
        del self.value[state]
