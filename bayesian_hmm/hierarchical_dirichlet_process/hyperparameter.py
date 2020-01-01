"""Hyperparameters combine a prior sampling function and a likelihood function."""
import copy
import typing

import numpy

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

    def log_likelihood(self):
        """The likelihood of the current parameter value."""
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
