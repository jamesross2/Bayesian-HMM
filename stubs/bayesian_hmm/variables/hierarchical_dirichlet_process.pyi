import typing
from . import auxiliary_variable as auxiliary_variable, dirichlet_process_family as dirichlet_process_family, hyperparameter as hyperparameter, states as states, stick_breaking_process as stick_breaking_process, variable as variable
from _typeshed import Incomplete

class HierarchicalDirichletProcess(variable.Variable):
    sticky: Incomplete
    alpha: Incomplete
    gamma: Incomplete
    kappa: Incomplete
    beta: Incomplete
    auxiliary_variable: Incomplete
    pi: Incomplete
    def __init__(self, sticky: bool, alpha: hyperparameter.Hyperparameter, gamma: hyperparameter.Hyperparameter, kappa: hyperparameter.Hyperparameter) -> None: ...
    def log_likelihood(self) -> float: ...
    def resample(self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]) -> None: ...
    def add_state(self, state: states.State) -> None: ...
    def remove_state(self, state: states.State) -> None: ...
