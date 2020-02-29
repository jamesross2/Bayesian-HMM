"""A non-parametric Bayesian hierarchical Dirichlet process.

This model ties together multiple Bayesian variables to create a single process. It is driven by two hyperparameters,
which govern a (non-parametric) Dirichlet process, which in turns forms a prior for a child Dirichlet process. The
child Dirichlet process forms a conjugate prior for use in the non-parametric Bayesian Hidden Markov model,
specifically as a prior / posterior pair for state transition probabilities.
"""

import typing

from . import auxiliary_variable, dirichlet_process, hyperparameter, states, stick_breaking_process, variable


class HierarchicalDirichletProcess(variable.Variable):
    """A non-parametric Bayesian hierarchical Dirichlet process."""

    def __init__(
        self,
        sticky: bool = False,
        alpha: hyperparameter.Hyperparameter = hyperparameter.Gamma(shape=2, scale=2),
        gamma: hyperparameter.Hyperparameter = hyperparameter.Gamma(shape=3, scale=3),
        kappa: typing.Optional[hyperparameter.Hyperparameter] = hyperparameter.Beta(shape=1, scale=1),
    ) -> None:
        """A non-parametric Bayesian hierarchical Dirichlet process.

        Args:
            sticky: If False (the default), the usual model is created. If True, then an additional Hyperparameter
                `kappa` is created, which governs the self-transition probabilities within the transition probabilities.
            alpha: the Hyperparameter controlling the variability of the stick breaking process.
            gamma: the Hyperparameter controlling the sampling of new states in the Dirichlet stick breaking process.
            kappa: the Hyperparameter controlling the stickyness of the hierarchical Dirichlet process. If sticky is
                False (the default), then this value is ignored.

        Raises:
            ValueError: If the chain is sticky but not kappa hyperparameter is provided.

        """
        # init parent
        super(HierarchicalDirichletProcess, self).__init__()

        # replace kappa with None if the process is not sticky
        if sticky and kappa is None:
            raise ValueError("Hyperparameter kappa must be given for a sticky process.")
        self.sticky: bool = sticky
        self.kappa: typing.Optional[hyperparameter.Hyperparameter] = kappa if self.sticky else None

        # store hyperparameters
        self.alpha: hyperparameter.Hyperparameter = alpha
        self.gamma: hyperparameter.Hyperparameter = gamma

        # create all component distributions (beta, auxiliary variables, and final HDP)
        self.beta: stick_breaking_process.StickBreakingProcess
        self.beta = stick_breaking_process.StickBreakingProcess(alpha=self.alpha)
        self.auxiliary_variable: auxiliary_variable.AuxiliaryVariable
        self.auxiliary_variable = auxiliary_variable.AuxiliaryVariable(alpha=self.alpha, beta=self.beta)
        self.pi: dirichlet_process.DirichletProcessFamily
        self.pi = dirichlet_process.DirichletProcessFamily(beta=self.beta, gamma=self.gamma, kappa=self.kappa)

    def log_likelihood(self) -> float:
        """The total log likelihood of the model, calculated as the sum of its component log likelihoods.

        Returns:
            The log likelihood of all parameters within the model (alpha, gamma, beta, pi, and kappa (if the model is
                sticky).

        """
        likelihoods = (
            self.alpha.log_likelihood(),
            self.gamma.log_likelihood(),
            self.beta.log_likelihood(),
            self.pi.log_likelihood(),
            self.kappa.log_likelihood() if self.sticky else 0.0,
        )
        return sum(likelihoods)

    def resample(self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]) -> None:
        """Performs one iteration of sampling for all variables within the Model.

        Resampling is performed for hyperparameters first(alpha, gamma, and kappa if the Model is sticky), then for the
        stick breaking process (the auxiliary variables and beta), and finally for the dirichlet process (pi). Each
        variable implements its own resampling process, but in general Hyperparameters use a Metropolis Hastings
        resampling approach, while the other variables use a Gibbs conditional resampling step.

        Args:
            counts: The number of transitions between states. The value of `counts[state0][state1]` should be the number
                of transitions from `state0` to `state1` in the latent variable series.

        """
        # fill in default values
        states_from = tuple(counts.keys())
        if len(states_from) == 0:
            states_to = set()
        else:
            states_to = set(counts[states_from[0]].keys())

        # begin by updating hyperparameters
        self.alpha.resample(posterior_log_likelihood=lambda x: self.beta.log_likelihood())
        self.gamma.resample(posterior_log_likelihood=lambda x: self.pi.log_likelihood())
        if self.sticky:
            self.kappa.resample(posterior_log_likelihood=lambda x: self.pi.log_likelihood())

        # next, auxiliary variables require beta to have correct values
        states_remove = self.beta.value.keys() - states_to
        for state in states_remove:
            self.beta.remove_state(state)

        states_add = states_to - self.beta.value.keys()
        for state_to in states_add:
            if state_to not in self.beta.value.keys():
                self.beta.add_state(state_to)

        # update beta (stick breaking process), which also requires updated auxiliary variables
        # first, ensure that stick breaking process has the correct states
        self.auxiliary_variable.resample(counts=counts, exact=True)
        self.beta.resample(auxiliary=self.auxiliary_variable)

        # finally, update pi (dirichlet process)
        self.pi.resample(counts=counts)

    def add_state(self, state: states.State) -> None:
        """Appends another state to the Dirichlet process.

        Args:
            state: The new state to include.

        """
        self.beta.add_state(state)
        self.pi.add_state(state)

    def remove_state(self, state: states.State) -> None:
        """Drops a state from the Dirichlet family; both its own transitions and the transitions into it.

        Args:
            state: The state to remove.

        """
        self.beta.remove_state(state=state)
        self.pi.remove_state(state=state)
