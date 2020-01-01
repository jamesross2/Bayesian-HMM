"""A non-parametric Bayesian hierarchical Dirichlet process.

This model ties together multiple Bayesian variables to create a single process. It is driven by two hyperparameters,
which govern a (non-parametric) Dirichlet process, which in turns forms a prior for a child Dirichlet process. The
child Dirichlet process forms a conjugate prior for use in the non-parametric Bayesian Hidden Markov model,
specifically as a prior / posterior pair for state transition probabilities.
"""

import typing

import scipy.stats

from . import auxiliary_variable, dirichlet_process, hyperparameter, stick_breaking_process, symbol, variable


class Model(variable.Variable):
    """A non-parametric Bayesian hierarchical Dirichlet process."""

    def __init__(
        self,
        alpha_prior: typing.Callable[[], float] = lambda: scipy.stats.gamma.rvs(a=2, scale=2),
        alpha_log_likelihood: typing.Callable[[float], float] = lambda x: scipy.stats.gamma.logpdf(x=x, a=2, scale=2),
        gamma_prior: typing.Callable[[], float] = lambda: scipy.stats.gamma.rvs(a=3, scale=3),
        gamma_log_likelihood: typing.Callable[[float], float] = lambda x: scipy.stats.gamma.logpdf(x=x, a=3, scale=3),
        sticky: bool = False,
        kappa_prior: typing.Callable[[], float] = lambda: scipy.stats.beta.rvs(a=1, b=1),
        kappa_log_likelihood: typing.Callable[[float], float] = lambda x: scipy.stats.beta.logpdf(x=x, a=1, b=1),
    ) -> None:
        """A non-parametric Bayesian hierarchical Dirichlet process.

        Args:
            alpha_prior: The prior distribution of alpha.
            alpha_log_likelihood: The prior log likelihood of alpha. Note that this is different to the likelihood
                function passed to (and contained in) the `Model.alpha` `Hyperparameter`, since that function is the
                posterior log likelihood for alpha.
            gamma_prior: Same as for alpha.
            gamma_log_likelihood: Same as for alpha.
            sticky: If False (the default), the usual model is created. If True, then an additional Hyperparameter
                `kappa` is created, which governs the self-transition probabilities within the transition probabilities.
            kappa_prior: Same as for alpha. If `sticky` is False, then this argument is ignored. The domain should be
                restricted to the interval `[0, 1]` (with `0` indicating no stickiness, and `1` being large stickiness).
            kappa_log_likelihood: Same as for alpha. If `sticky` is False, then this argument is ignored.

        """
        # init parent
        super(Model, self).__init__()

        # note whether model should be made sticky or not
        self.sticky: bool = sticky

        # create hyperparameters for alpha and gamma
        self.alpha: hyperparameter.Hyperparameter = hyperparameter.Hyperparameter(alpha_prior, alpha_log_likelihood)
        self.gamma: hyperparameter.Hyperparameter = hyperparameter.Hyperparameter(gamma_prior, gamma_log_likelihood)
        self.kappa: typing.Optional[hyperparameter.Hyperparameter]
        self.kappa = hyperparameter.Hyperparameter(kappa_prior, kappa_log_likelihood) if self.sticky else None

        # create stick breaking process
        self.beta = stick_breaking_process.StickBreakingProcess(alpha=self.alpha)
        self.auxiliary_variable = auxiliary_variable.AuxiliaryVariable(alpha=self.alpha, beta=self.beta)

        # create child dirichlet process
        self.pi = dirichlet_process.DirichletProcess(beta=self.beta, gamma=self.gamma, kappa=self.kappa)

        # store stores
        self.states: typing.Set[symbol.Symbol] = {symbol.EmptySymbol()}

    def log_likelihood(self):
        """The total log likelihood of the model, calculated as the sum of its component log likelihoods."""
        likelihoods = (
            self.alpha.log_likelihood(),
            self.gamma.log_likelihood(),
            self.beta.log_likelihood(),
            self.pi.log_likelihood(),
        )
        return sum(likelihoods)

    def resample(
        self, states: typing.Set[symbol.Symbol], counts: typing.Dict[symbol.Symbol, typing.Dict[symbol.Symbol, int]]
    ) -> None:
        """Performs one iteration of sampling for all variables within the Model.

        Resampling is performed for hyperparameters first(alpha, gamma, and kappa if the Model is sticky), then for the
        stick breaking process (the auxiliary variables and beta), and finally for the dirichlet process (pi). Each
        variable implements its own resampling process, but in general Hyperparameters use a Metropolis Hastings
        resampling approach, while the other variables use a Gibbs conditional resampling step.

        Args:
            states: A set of observed states. After the function is complete, the stick breaking process and the
                Dirichlet process will both contain only these states, and the usual ``EmptyState``.
            counts: The number of transitions between states. The value of `counts[state0][state1]` should be the number
                of transitions from `state0` to `state1` in the latent variable series.

        """
        # first, ensure that all states exist in the stick breaking process
        for state in states:
            if state not in self.beta.value.keys():
                self.beta.add_state(state)

        # begin by updating hyperparameters
        self.alpha.resample(posterior_log_likelihood=lambda x: self.beta.log_likelihood())
        self.gamma.resample(posterior_log_likelihood=lambda x: self.pi.log_likelihood())
        if self.sticky:
            self.kappa.resample(posterior_log_likelihood=lambda x: self.pi.log_likelihood())

        # next, update beta (stick breaking process), which also requires updated auxiliary variables
        self.auxiliary_variable.resample(states=states, counts=counts, exact=True)
        self.beta.resample(states=states, auxiliary=self.auxiliary_variable)

        # finally, update pi (dirichlet process)
        self.pi.resample(states=states, counts=counts)

    def add_state(self, state: symbol.Symbol) -> None:
        """Appends another state to the Dirichlet process.

        Args:
            state: The state to be added to the Model.

        """
        self.beta.add_state(state)
        self.pi.add_state(state)

    def remove_state(self, state: symbol.Symbol) -> None:
        raise NotImplementedError("TODO: allow states to be removed to stick breaking process")
