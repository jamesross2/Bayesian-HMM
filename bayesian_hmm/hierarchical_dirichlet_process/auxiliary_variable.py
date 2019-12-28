from . import symbol
from . import hyperparameter
from . import variable
import typing
import numpy
import scipy.stats
import scipy.special
import sympy.functions.combinatorial.numbers


class AuxiliaryVariable(variable.Variable):
    def __init__(
        self,
        alpha: hyperparameter.Hyperparameter,
        beta: variable.Variable,
        observations: variable.Variable,
    ) -> None:
        # init parent
        super(variable.Variable, self).__init__()

        # auxiliary variables combine multiple variables (used to remove dependence in beta child)
        self.alpha: hyperparameter.Hyperparameter = alpha
        self.beta: variable.Variable = beta
        self.observations: variable.Variable = observations

        # fill with initial value
        self.value: typing.Dict[
            symbol.Symbol, typing.Dict[symbol.Symbol, int]
        ] = self.resample(states=[])

    @staticmethod
    def single_variable_log_likelihood(
        scale: float, value: int, count: int, exact: bool
    ):
        if exact:
            value = (
                numpy.log(scipy.special.gamma(scale))
                - numpy.log(scipy.special.gamma(scale + count))
                + value * numpy.log(scale)
                + numpy.log(
                    sympy.functions.combinatorial.numbers.stirling(count, value, kind=1)
                )
            )
        else:
            value = (
                value
                + (value + scale - 0.5) * numpy.log(scale)
                + (value - 1) * numpy.log(0.57721 + numpy.log(count - 1))
                - (value - 0.5) * numpy.log(value)
                - scale * numpy.log(scale + count)
                - scale
            )
        return value

    def single_variable_resample(
        self, state0: symbol.Symbol, state1: symbol.Symbol, exact: bool
    ) -> int:
        # TODO: check if beta[state0] or beta[state1]
        scale = self.alpha.value * self.beta.value[state0]
        count = self.observations.count[state0][state1]

        # initialise values required to resample
        p_required = numpy.random.uniform(0, 1)
        value_proposed = 0
        p_cumulative = 0

        # increase m iteratively, until required probability is met
        while p_cumulative < p_required and value_proposed < count:
            value_proposed += 1
            try:
                p_cumulative += numpy.exp(
                    AuxiliaryVariable.single_variable_log_likelihood(
                        scale=scale, value=value_proposed, count=count, exact=exact
                    )
                )
            except (RecursionError, OverflowError):
                # avoid using exact value in the future (errors almost certain to reoccur)
                exact = False
                p_cumulative += numpy.exp(
                    AuxiliaryVariable.single_variable_log_likelihood(
                        scale=scale, value=value_proposed, count=count, exact=exact
                    )
                )

        # if successful, sufficient probability accumulated--take maximum to guarantee non-zero value
        return max(value_proposed, 1)

    def log_likelihood(
        self, states: typing.Sequence[symbol.Symbol], exact: bool = True
    ) -> float:
        # TODO: check if beta[state0] or beta[state1]
        # TODO: check if observations[state0] or observations[state1]
        # TODO: ensure that summation performed properly
        log_likelihoods = {
            state0: numpy.sum(
                self.single_variable_log_likelihood(
                    scale=self.alpha.value * self.beta.value[state0],
                    value=self.value[state0][state1],
                    count=self.observations[state0][state1],
                )
                for state1 in states
            )
            for state0 in states
        }
        return sum(log_likelihoods.values())

    def resample(
        self, states: typing.Sequence[symbol.Symbol], exact: bool = True
    ) -> typing.Dict[symbol.Symbol, typing.Dict[symbol.Symbol, int]]:
        # sample individual variables to begin
        self.value = {
            state0: {
                state1: self.single_variable_resample(state0, state1)
                for state1 in states
            }
            for state0 in states
        }
        return self.value

    @property
    def aggregated_value(self) -> typing.Dict[symbol.Symbol, int]:
        # TODO: determine if this property is required
        states = self.value.keys()
        return {
            state1: sum(self.value[state0][state1] for state0 in states)
            for state1 in states
        }
