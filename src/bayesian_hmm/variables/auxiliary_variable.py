"""The auxiliary variables are used to capture the conditional dependence of the stick breaking process."""

import typing
import warnings

import numpy
import scipy.special
import scipy.stats
import sympy.functions.combinatorial.numbers

from . import hyperparameter, states, stick_breaking_process, variable


class AuxiliaryVariable(variable.Variable):
    def __init__(self, alpha: hyperparameter.Hyperparameter, beta: stick_breaking_process.StickBreakingProcess) -> None:
        """The auxiliary variables parametrise the posterior distribution of the stick breaking process.

        The 'auxiliary variable' is a standalone variable used to make the conditional distribution
        of beta (the stick breaking process) more straightforward. This greatly simplifies the
        resampling steps of the hierarchical Dirichlet process.

        The auxiliary variable combines information from the alpha hyperparameter, the current value
        of beta, and of the transition counts. The actual distribution is complex; refer to the
        hierarchical Dirichlet process documentation for sources.

        Args:
            alpha: The Hyperparameter governing the stick breaking process in the hierarchical Dirichlet
                process.
            beta: The stick breaking process itself.

        """
        # init parent
        super(AuxiliaryVariable, self).__init__()

        # auxiliary variables combine multiple variables (used to remove dependence in beta child)
        self.alpha: hyperparameter.Hyperparameter = alpha
        self.beta: stick_breaking_process.StickBreakingProcess = beta

        # fill with empty initial value
        self.value: typing.Dict[states.State, typing.Dict[states.State, int]] = {}

    @staticmethod
    def single_variable_log_likelihood(scale: float, value: int, count: int, exact: bool) -> float:
        """The posterior likelihood of a single auxiliary variable element.

        Args:
            scale: A parameter of the distribution equal to alpha * beta (beta for the second state
                of interest)
            value: The current value of the variable
            count: The observed transition counts from the first state to the second state.
            exact: If True, computes the log likelihood of the atomic variable up to machine error.
                If False, uses an approximation (much faster for large values of 'count', but more
                inaccurate for high values of 'count').

        Returns:
            Posterior log likelihood of the given value.

        """
        # sympy returns 'number' objects that print as numeric but throw obscure errors
        if exact:
            log_likelihood = (
                numpy.log(scipy.special.gamma(scale))
                - numpy.log(scipy.special.gamma(scale + count))
                + value * numpy.log(scale)
                + numpy.log(float(sympy.functions.combinatorial.numbers.stirling(count, value, kind=1)))
            )
        else:
            log_likelihood = (
                value
                + (value + scale - 0.5) * numpy.log(scale)
                + (value - 1) * numpy.log(0.57721 + numpy.log(count - 1))
                - (value - 0.5) * numpy.log(value)
                - scale * numpy.log(scale + count)
                - scale
            )
        return log_likelihood

    @staticmethod
    def single_variable_resample(scale: float, count: int, exact: bool) -> int:
        """Sample a single auxiliary variable with given distribution.

        If either a RecursionError or OverflowError are raised while calculating an exact log likelihood (required for
        resampling), then the approximation will be used instead.

        Args:
            scale: Equal to alpha * beta for the second state of interest.
            count: The number of observed transition between two states of interest.
            exact: A flag to indicate whether sampling should be performed exactly or approximately.

        Returns:
            An auxiliary variable, sampled with conditional distribution.

        Raises:
            RecursionError: If an error is raised while computing the approximate log likelihood that is not recognised, it is
                passed on to the user.
            OverflowError: If an error is raised while computing the approximate log likelihood that is not recognised, it is
                passed on to the user.
        """
        # TODO: check if we can simplify this
        if count <= 0:
            return 1

        # initialise values required to resample
        p_required = numpy.random.uniform(0, 1)
        value_proposed = 0
        p_cumulative = 0

        # increase m iteratively, until required probability is met
        while p_cumulative < p_required and value_proposed < count:
            # add probability for next state to accumulated probability
            value_proposed += 1
            try:
                p_cumulative += numpy.exp(
                    AuxiliaryVariable.single_variable_log_likelihood(
                        scale=scale, value=value_proposed, count=count, exact=exact
                    )
                )
            except (RecursionError, OverflowError) as err:
                # avoid using exact value in the future (errors almost certain to reoccur)
                if exact:
                    # exact method failed, resort to approximation and restart current value
                    exact = False
                    value_proposed -= 1
                else:
                    # approximation failed, return error
                    raise err

        # if successful, sufficient probability accumulated--take maximum to guarantee non-zero value
        return value_proposed

    def log_likelihood(
        self, counts: typing.Dict[states.State, typing.Dict[states.State, int]], exact: bool = True
    ) -> float:
        """Calculate the log likelihood of all auxiliary variables.

        Note that this log likelihood differs to the likelihoods of other Bayesian variables in the hierarchical
        Dirichlet process, since they are not actually a parameter of the model (and not subject to the usual prior /
        posterior laws of such a variable). Instead, they are used to simplify the resampling step of the stick breaking
        process, which is conditionally independent given its auxiliary variable.

        Args:
            counts: The number of transitions of each type. Note
            exact: If True (the default), computes the exact likelihood whenever possible.

        Returns:
            The sum of log likelihoods of each auxiliary variable.

        """
        warnings.warn("Calculated likelihood of auxiliary variables should not contribute to model likelihood.")

        # fill in default values
        states_from = tuple(counts.keys())
        states_to: typing.Tuple
        if len(states_from) == 0:
            states_to = tuple()
        else:
            states_to = tuple(counts[states_from[0]].keys())

        # TODO: check if beta[state_from] or beta[state_to]
        # TODO: ensure that summation performed properly
        log_likelihoods = {
            state_from: sum(
                self.single_variable_log_likelihood(
                    scale=self.alpha.value * self.beta.value[state_to],
                    value=self.value[state_from][state_to],
                    count=counts[state_from][state_to],
                    exact=exact,
                )
                for state_to in states_to
            )
            for state_from in states_from
        }
        return sum(log_likelihoods.values())

    # TODO: add multiprocessing abilities to resampling step here
    def resample(
        self, counts: typing.Dict[states.State, typing.Dict[states.State, int]], exact: bool = True
    ) -> typing.Dict[states.State, typing.Dict[states.State, int]]:
        """Fill the value attribute of the AuxiliaryVariable with new values according to the marginal distribution.

        Args:
            counts: The transition counts between states for the current set of latent variables.
            exact: If True (the default), computes the exact likelihood whenever possible.

        Returns:
            The resampled value.

        """
        # fill in default values
        states_from = tuple(counts.keys())
        states_to: typing.Tuple
        if len(states_from) == 0:
            states_to = tuple()
        else:
            states_to = tuple(counts[states_from[0]].keys())

        # TODO: check if beta[state_from] or beta[state_to]
        value = {
            state_from: {
                state_to: self.single_variable_resample(
                    scale=self.alpha.value * self.beta.value[state_to], count=counts[state_from][state_to], exact=exact
                )
                for state_to in states_to
            }
            for state_from in states_from
        }
        self.value = value
        return self.value

    def value_aggregated(self) -> typing.Dict[states.State, int]:
        """The AuxiliaryVariables after the aggregation required to resample the stick breaking process.

        This is a convenience function only, since the stick breaking process uses sums of auxiliary variables.

        Returns:
            Contains the sum of the auxiliary variables for each state.

        """
        # fill in default values
        states_from = tuple(self.value.keys())
        states_to: typing.Tuple
        if len(states_from) == 0:
            states_to = tuple()
        else:
            states_to = tuple(self.value[states_from[0]].keys())

        return {state_to: sum(self.value[state_from][state_to] for state_from in states_from) for state_to in states_to}
