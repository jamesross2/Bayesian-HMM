from . import symbol
from . import variable
from . import hyperparameter
from . import auxiliary_variable
import typing
import scipy.stats
import scipy.special


class StickBreakingProcess(variable.Variable):
    def __init__(
        self,
        alpha: hyperparameter.Hyperparameter
    ):
        # init parent
        super(variable.Variable, self).__init__()

        # parent information is the alpha hyperparameter and auxiliary hyperparameters
        self.alpha: hyperparameter.Hyperparameter = alpha

        # fill with initial value
        self.value: typing.Dict[symbol.Symbol, float] = self.resample(
            states=[],
            auxiliary_variable=auxiliary_variable.AuxiliaryVariable(alpha, [], [])
        )  # redundant assignment to avoid errors

    def log_likelihood(self) -> float:
        # calculated as a Dirichlet variable with a special aggregate state
        return scipy.stats.dirichlet.logpdf(self.value.values(), self.alpha.value)

    def resample(
        self,
        states: typing.Sequence[symbol.Symbol],
        auxiliary_variable: auxiliary_variable.AuxiliaryVariable
    ) -> typing.Dict[symbol.Symbol, float]:
        # extract aggregate auxiliary variables to get parameters of conditional distribution for beta
        parameters = {
            s2: sum(auxiliary_variable.value[s1][s2] for s1 in states)
            for s2 in states
        }

        # resample using parameters of conditional distribution
        values = scipy.stats.dirichlet.rvs(alpha=parameters, size=1)

        # associate values back to states
        self.value = dict(zip(states, values))
        return self.value

    def sample_auxiliary_variables(
        self, states: typing.Sequence[symbol.Symbol]
    ) -> typing.Dict[symbol.Symbol, float]:
        # TODO: fix! what do aux variables point to in the conditional?
        auxiliary_variables = {
            s1: {
                s2: auxiliary_variable.AuxiliaryVariable.resample(
                    scale=self.alpha.value * self.value[s2], n=1
                )
                for s2 in states
            }
            for s1 in states
        }
        parameters = {
            s2: sum(auxiliary_variables[s1][s2] for s1 in states) for s2 in states
        }
        return parameters

    def add_state(self, symbol: symbol.Symbol) -> None:
        pass

    def remove_state(self, symbol: symbol.Symbol) -> None:
        pass
