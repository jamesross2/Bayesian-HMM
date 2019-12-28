"""A Bayesian dirichlet distribution."""

from . import variable
import typing
import numpy


class Hyperparameter(variable.Variable):
    def __init__(
        self,
        prior: typing.Callable[[], typing.Union[float, int]],
        log_likelihood: typing.Callable[[typing.Union[float, int]], float],
    ) -> None:
        super(variable.Variable, self).__init__()

        # parent information replaced by prior & likelihood for hyperparameter
        self.prior = prior
        self._log_likelihood_base = log_likelihood

        # fill with initial value
        self.value: typing.Union[float, int] = self.prior()

    def log_likelihood(self):
        return self._log_likelihood_base(self.value)

    def resample(self, force: bool = False) -> typing.Union[float, int]:
        # if forcing, do not apply Metropolis Hastings algorithm
        if force:
            self.value = self.prior()
            return self.value

        # Metropolis Hastings resampling compares current likelihood to proposed likelihood
        value_proposed = self.prior()
        log_likelihood_proposed = self._log_likelihood_base(value_proposed)

        # choose whether to accept using standard Metropolis Hasting identity
        accept_probability = min(
            1, numpy.exp(log_likelihood_proposed - self.log_likelihood())
        )
        accepted = bool(numpy.random.binomial(n=1, p=accept_probability))

        # update hyperparameter value
        if accepted:
            self.value = value_proposed

        return self.value
