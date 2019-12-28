import bayesian_hmm
from . import symbol
from . import variable
import typing
import numpy


class BayesianDirichletProcess(variable.Variable):
    def __init__(
        self,
        beta: variable.Variable,
        gamma: variable.Variable,
        states: variable.Variable,
        sticky: bool = False,
    ):
        # init parent
        super(variable.Variable, self).__init__()

        # store parents
        self.beta: variable.Variable = beta
        self.gamma: variable.Variable = gamma
        self.states: variable.Variable = states
        self.sticky = sticky

        # fill current value
        self.value = self.resample()

    def log_likelihood(self) -> float:
        # TODO
        return 1

    def resample(self) -> typing.Dict[symbol.Symbol, float]:
        pass

    def get_parameters(self):
        if self.sticky:
            params = {
                s2: self.n_transition[state][s2]
                + self.hyperparameters["alpha"]
                * (1 - self.hyperparameters["kappa"])
                * self.beta_transition[s2]
                for s2 in self.states
            }
            params[bayesian_hmm.EmptySymbol()] = (
                self.hyperparameters["alpha"]
                * (1 - self.hyperparameters["kappa"])
                * self.beta_transition[bayesian_hmm.EmptySymbol()]
            )
            params[state] += (
                self.hyperparameters["alpha"] * self.hyperparameters["kappa"]
            )
        else:
            params = {
                s2: self.n_transition[state][s2]
                + self.hyperparameters["alpha"] * self.beta_transition[s2]
                for s2 in self.states
            }
            params[bayesian_hmm.EmptySymbol()] = (
                self.hyperparameters["alpha"]
                * self.beta_transition[bayesian_hmm.EmptySymbol()]
            )

        return params

    def resample_p_transition(self, eps=1e-12):
        """
        Resample the transition probabilities from the current beta values and kappa
        value, if the chain is sticky.
        :param eps: minimum expected value passed to Dirichlet sampling step
        :return: None
        """
        # empty current transition values
        self.p_transition = {}

        # refresh each state in turn
        for state in self.states:
            params = self._get_p_transition_metaparameters(state)
            temp_result = np.random.dirichlet(list(params.values())).tolist()
            p_transition_state = dict(zip(list(params.keys()), temp_result))
            self.p_transition[state] = shrink_probabilities(p_transition_state, eps)

        # add transition probabilities from unseen states
        # note: no stickiness update because these are aggregated states
        params = {
            k: self.hyperparameters["alpha"] * v
            for k, v in self.beta_transition.items()
        }
        temp_result = np.random.dirichlet(list(params.values())).tolist()
        p_transition_none = dict(zip(list(params.keys()), temp_result))
        self.p_transition[bayesian_hmm.EmptySymbol()] = shrink_probabilities(
            p_transition_none, eps
        )

    def calculate_p_transition_loglikelihood(self):
        """
        Note: this calculates the likelihood over all entries in the transition matrix.
        If chains have been resampled (this is the case during MCMC sampling, for
        example), then there may be entries in the transition matrix that no longer
        correspond to actual states.
        :return:
        """
        ll_transition = 0
        states = self.p_transition.keys()

        # get probability for each state
        for state in states:
            params = self._get_p_transition_metaparameters(state)
            p_transition = shrink_probabilities(
                {s: self.p_transition[state][s] for s in states}
            )
            ll_transition += np.log(
                stats.dirichlet.pdf(
                    [p_transition[s] for s in states], [params[s] for s in states]
                )
            )

        # get probability for aggregate state
        params = {
            k: self.hyperparameters["alpha"] * v
            for k, v in self.beta_transition.items()
        }
        ll_transition += np.log(
            stats.dirichlet.pdf(
                [self.p_transition[bayesian_hmm.EmptySymbol()][s] for s in states],
                [params[s] for s in states],
            )
        )

        return ll_transition


class DirichletProcess(object):
    """A Bayesian Dirichlet distribution, with prior and posterior probabilities."""

    def __init__(self, alpha: float, epsilon: float = 1e-12) -> None:
        # save inputs
        self.alpha = alpha
        self.epsilon = epsilon

    def likelihood(self, beta: typing.Sequence[float]) -> float:
        raise NotImplementedError("TODO")

    def sample_beta(
        self, counts: typing.Dict[symbol.Symbol, int]
    ) -> typing.Dict[symbol.Symbol, float]:
        params = {
            s: self.n_initial[s]
            + self.hyperparameters["alpha"] * self.beta_transition[s]
            for s in self.states
        }
        params[bayesian_hmm.EmptySymbol()] = (
            self.hyperparameters["alpha"]
            * self.beta_transition[bayesian_hmm.EmptySymbol()]
        )
        return params

    def resample_p_initial(self, eps=1e-12):
        """
        Resample the starting probabilities. Performed as a sample from the posterior
        distribution, which is a Dirichlet with pseudocounts and actual counts combined.
        :param eps: minimum expected value.
        :return: None.
        """
        params = self._get_p_initial_metaparameters()
        temp_result = np.random.dirichlet(list(params.values())).tolist()
        p_initial = dict(zip(list(params.keys()), temp_result))
        self.p_initial = shrink_probabilities(p_initial, eps)

    def calculate_p_initial_loglikelihood(self):
        params = self._get_p_initial_metaparameters()
        p_initial = shrink_probabilities(
            {state: self.p_initial[state] for state in params.keys()}
        )
        ll_initial = np.log(
            stats.dirichlet.pdf(
                [p_initial[s] for s in params.keys()],
                [params[s] for s in params.keys()],
            )
        )
        return ll_initial

    def sample_posterior(
        self, counts: typing.Sequence[int]
    ) -> typing.Dict[symbol.Symbol, float]:
        posterior_parameters = [
            self.alpha + beta * count for beta, count in zip(self.beta, counts)
        ]
        beta = numpy.random.dirichlet(alpha=posterior_parameters, size=1)
        return beta

    def likelihood(self):
        pass


class HierarchicalBayesianDirichletProcess(object):
    """A family of Bayesian Dirichlet distributions with shared parameters."""

    def __init__(
        self,
        alpha_prior: typing.Callable[[], float],
        gamma_prior: typing.Callable[[], float],
    ) -> None:
        # save inputs
        self.alpha_prior = alpha_prior
        self.gamma_prior = gamma_prior

        # store child Dirichlet distributions in another dictionary
        self.children: typing.Dict[symbol.Symbol, DirichletProcess] = dict()

        # initialise hyperparameter values
        self.alpha = self.sample_alpha_prior()
        self.gamma = self.sample_gamma_prior()

        # initialise parameter values
        self.beta: DirichletProcess = DirichletProcess(alpha=self.alpha)
        self.auxiliary_vars: typing.Dict[symbol.Symbol, float] = {}

    def sample_alpha_prior(self) -> float:
        self.alpha = self.alpha_prior()
        return self.alpha

    def sample_alpha_posterior(self) -> float:
        # find current likelihood & remember alpha
        likelihood_curr = self.beta.likelihood(self.alpha)
        alpha_curr = self.alpha

        # sample a new alpha and apply MH testing to resample
        self.sample_alpha_prior()
        likelihood_proposed = self.beta.likelihood(self.alpha)

    def sample_gamma_prior(self) -> float:
        self.gamma = self.gamma_prior()
        return self.gamma

    def sample_dirichlet_posteriors(
        self, counts: typing.Dict[symbol.Symbol, float]
    ) -> typing.Dict[symbol.Symbol, DirichletProcess]:
        """Resamples the child  distributions."""
        self.children = {
            symbol: DirichletProcess(counts[symbol]) for symbol in counts.keys()
        }
        return self.children
