"""Hyperparameters combine a prior sampling function and a likelihood function."""

import copy
import typing

import numpy
import scipy.stats

from . import variable


class Hyperparameter(variable.Variable):
    def __init__(
        self,
        prior: typing.Callable[[], typing.Union[float, int]],
        log_likelihood: typing.Callable[[typing.Union[float, int]], float],
    ) -> None:
        """A generic hyperparameter for a Bayesian model.

        Args:
            prior: A prior distribution for the parameter of interest. Calls to this function should return new,
                unconditional samples of the hyperparameter.
            log_likelihood: A function that calculates the unconditional log likelihood of the variable of
                interest; i.e. the prior distribution pdf. This will be used to calculate the likelihood of the model.

        """
        super(Hyperparameter, self).__init__()

        # parent information replaced by prior & likelihood for hyperparameter
        self.prior = prior
        self.prior_log_likelihood = log_likelihood

        # fill with initial value
        self.value: typing.Union[float, int] = self.prior()

    def log_likelihood(self) -> float:
        """The likelihood of the current parameter value.

        Returns:
            The log likelihood of the parameter's current value.

        """
        return self.prior_log_likelihood(self.value)

    def resample(
        self,
        posterior_log_likelihood: typing.Optional[typing.Callable[[typing.Union[float, int]], float]] = None,
        force: bool = False,
    ) -> typing.Union[float, int]:
        """Use Metropolis Hastings to resample the hyperparameter value.

        Args:
            posterior_log_likelihood: A function that calculates the conditional log likelihood of the variable of
                interest; i.e. the posterior distribution pdf. This will be used for Metropolis Hastings rejection
                sampling, so typically this function will incorporate information about dependent parameters.
            force: If False (the default), Metropolis Hastings rejection sampling will be used to resample the
                hyperparameter value. If True, a new value is taken from the `prior` attribute and no rejection
                sampling is applied.

        Returns:
            The new hyperparameter value.

        Raises:
            ValueError: The `posterior_log_likelihood` must be given when `force` is False.

        """
        # if forcing, do not apply Metropolis Hastings algorithm
        if force:
            self.value = self.prior()
            return self.value

        if not force and posterior_log_likelihood is None:
            raise ValueError("Posterior likelihood required if force is False.")
        assert posterior_log_likelihood is not None  # stop mypy errors

        # Metropolis Hastings resampling compares current likelihood to proposed likelihood
        value_initial = copy.deepcopy(self.value)
        log_likelihood_initial = posterior_log_likelihood(self.value)
        self.value = self.prior()
        log_likelihood_proposed = posterior_log_likelihood(self.value)

        # choose whether to accept using standard Metropolis Hasting form
        accept_probability = numpy.exp(min(0.0, log_likelihood_proposed - log_likelihood_initial))
        accepted = bool(numpy.random.binomial(n=1, p=accept_probability))
        if not accepted:
            self.value = value_initial

        return self.value


# common gamma Hyperparameter
class Gamma(Hyperparameter):
    def __init__(self, shape: float = 1, scale: float = 1) -> None:
        """A Gamma-distributed hyperparameter for a Bayesian model.

        Args:
            shape: the shape parameter of the Gamma distribution
            scale: the scale parameter of the Gamma distribution

        """
        # get gamma prior and log likelihood functions
        prior: typing.Callable[[], float] = lambda: scipy.stats.gamma.rvs(a=shape, scale=scale)
        log_likelihood: typing.Callable[[float], float] = lambda x: scipy.stats.gamma.logpdf(x=x, a=shape, scale=scale)
        super(Gamma, self).__init__(prior, log_likelihood)


# common beta Hyperparameter
class Beta(Hyperparameter):
    def __init__(self, shape: float = 1, scale: float = 1) -> None:
        """A Beta-distributed hyperparameter for a Bayesian model.

        Args:
            shape: the shape parameter of the Beta distribution
            scale: the scale parameter of the Beta distribution

        """
        # get gamma prior and log likelihood functions
        prior: typing.Callable[[], float] = lambda: scipy.stats.beta.rvs(a=shape, b=scale)
        log_likelihood: typing.Callable[[float], float] = lambda x: scipy.stats.beta.logpdf(x=x, a=shape, b=scale)
        super(Beta, self).__init__(prior, log_likelihood)


class Dummy(Hyperparameter):
    def __init__(self, value: typing.Union[int, float] = 0.0) -> None:
        """A Hyperparameter that only takes a single value.

        Args:
            value: the value assumed by the Hyperparameter at all times

        """
        # get gamma prior and log likelihood functions
        prior: typing.Callable[[], float] = lambda: value
        log_likelihood: typing.Callable[[float], float] = lambda x: 0.0
        super(Dummy, self).__init__(prior, log_likelihood)

    def resample(
        self,
        posterior_log_likelihood: typing.Optional[typing.Callable[[typing.Union[float, int]], float]] = None,
        force: bool = False,
    ) -> typing.Union[float, int]:
        """Simply return the dummy value of the Dummy Hyperparameter without any resampling.

        All arguments are ignored, and left only for signature consistency.

        Args:
            posterior_log_likelihood: Ignored.
            force: Ignored.

        Returns:
            The dummy value of the Hyperparameter.

        """
        return self.value
