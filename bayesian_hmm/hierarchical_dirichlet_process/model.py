"""A non-parametric Bayesian Hierarchical Dirichlet Process.

This model ties together multiple Bayesian variable to create a
single process. It is driven by two hyperparameters, which govern
a (non-parametric) Dirichlet process, which in turns forms a prior
for a child Dirichlet process. The child Dirichlet process forms a
conjugate prior for use in the non-parametric Bayesian Hidden
Markov model, specifically as a prior / posterior pair for state
transition probabilities.

# TODO: finish docstring.
"""

from . import variable
from . import hyperparameter
from . import stick_breaking_process
from . import dirichlet_process
import typing
import numpy
import scipy.stats
import warnings


class Model(variable.Variable):
    def __init__(
        self,
        alpha_prior: typing.Callable[[], typing.Union[float, int]] = lambda: numpy.random.gamma(2, 2),
        alpha_log_likelihood: typing.Callable[[typing.Union[float, int]], float] = scipy.stats.gamma.logpdf,
        gamma_prior: typing.Callable[[], typing.Union[float, int]] = lambda: numpy.random.gamma(2, 2),
        gamma_log_likelihood: typing.Callable[[typing.Union[float, int]], float] = scipy.stats.gamma.logpdf
    ) -> None:
        # init parent
        super(variable.Variable, self).__init__()

        # create hyperparameters for alpha and gamma
        self.alpha = hyperparameter.Hyperparameter(alpha_prior, alpha_log_likelihood)
        self.gamma = hyperparameter.Hyperparameter(gamma_prior, gamma_log_likelihood)

        # create stick breaking process
        self.beta = stick_breaking_process.StickBreakingProcess(alpha=self.alpha)

        # create child dirichlet process
        self.pi = dirichlet_process.DirichletProcess(alpha=self.alpha)

        # TODO: add 'sticky' capability
        # TODO: add resample capability
        warnings.warn("HDP implementation incomplete")

    def log_likelihood(self):
        likelihoods = (self.alpha.log_likelihood(), self.gamma.log_likelihood(), self.beta.log_likelihood(), self.pi.log_likelihood())
        return sum(likelihoods)
