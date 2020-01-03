"""A Bayesian Dirichlet process."""

import typing

import scipy.stats

from . import dirichlet_family, hyperparameter, states, variable


class HierarchicalDirichlet(variable.Variable):
    """A non-parametric Bayesian hierarchical Dirichlet process."""

    def __init__(
        self,
        beta_prior: typing.Callable[[], float] = lambda: scipy.stats.gamma.rvs(a=2, scale=2),
        beta_log_likelihood: typing.Callable[[float], float] = lambda x: scipy.stats.gamma.logpdf(x=x, a=2, scale=2),
    ) -> None:
        """A non-parametric Bayesian hierarchical Dirichlet process.

        Args:
            beta_prior: The prior distribution of beta, for a single categorical emission.
            beta_log_likelihood: The prior log likelihood of beta. Note that this is different to the likelihood
                function passed to (and contained in) the `Model.alpha` `Hyperparameter`, since that function is the
                posterior log likelihood for the dirichlet process.

        """
        # init parent
        super(HierarchicalDirichlet, self).__init__()

        # create hyperparameter for beta
        self.beta: hyperparameter.Hyperparameter
        self.beta = hyperparameter.Hyperparameter(prior=beta_prior, log_likelihood=beta_log_likelihood)

        # create a Dirichlet family governed by beta
        self.pi = dirichlet_family.DirichletFamily(beta=self.beta)

    def log_likelihood(self) -> float:
        """The total log likelihood of the model, calculated as the sum of its component log likelihoods.

        Returns:
            The log likelihood as a float. This is the sum of the beta and pi log likelihoods.
        """
        log_likelihoods = (self.beta.log_likelihood(), self.pi.log_likelihood())
        return sum(log_likelihoods)

    def resample(self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]) -> None:
        """Performs one iteration of sampling for all variables within the Model.

        Resampling is performed for hyperparameters first (beta), then for the then for the dirichlet process (pi). Each
        variable implements its own resampling process, but in general Hyperparameters use a Metropolis Hastings
        resampling approach, while the other variables use a Gibbs conditional resampling step.

        Args:
            counts: The number of transitions between states. The value of `counts[state0][state1]` should be the number
                of transitions from `state0` to `state1` in the emissions.

        """
        # begin by updating hyperparameters
        self.beta.resample(posterior_log_likelihood=lambda x: self.pi.log_likelihood())

        # update beta (stick breaking process), which also requires updated auxiliary variables
        self.pi.resample(counts=counts)

    def add_state(self, state: states.State, inner: bool = True, outer: bool = True) -> None:
        """Appends another state to the Dirichlet process.

        Args:
            state: The new state to include.
            inner: If True (the default), then the new symbol will be added to the inner keys for the Dirichlet family.
            outer: If True (the default), then the new symbol will be added to the outer keys for the Dirichlet family.

        """
        self.pi.add_state(state, inner=inner, outer=outer)

    def remove_state(self, state: states.State, inner: bool = True, outer: bool = True) -> None:
        """Drops a state from the Dirichlet family; both its own transitions and the transitions into it.

        Args:
            state: The state to remove.
            inner: If True (the default), then the new state will be removed from the inner keys for the Dirichlet
                family.
            outer: If True (the default), then the new state will be removed from the outer keys for the Dirichlet
                family.

        """
        self.pi.remove_state(state=state, inner=inner, outer=outer)
