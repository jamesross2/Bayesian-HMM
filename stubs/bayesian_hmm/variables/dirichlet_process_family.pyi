import typing
from . import hyperparameter as hyperparameter, states as states, stick_breaking_process as stick_breaking_process, variable as variable
from .. import utils as utils
from _typeshed import Incomplete

class DirichletProcessFamily(variable.Variable):
    beta: Incomplete
    gamma: Incomplete
    kappa: Incomplete
    value: Incomplete
    def __init__(self, beta: stick_breaking_process.StickBreakingProcess, gamma: hyperparameter.Hyperparameter, kappa: hyperparameter.Hyperparameter) -> None: ...
    @property
    def sticky(self) -> bool: ...
    @property
    def states_inner(self) -> typing.Set[states.State]: ...
    @property
    def states_outer(self) -> typing.Set[states.State]: ...
    def posterior_parameters(self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]) -> typing.Dict[states.State, typing.Dict[states.State, float]]: ...
    def log_likelihood(self) -> float: ...
    def resample(self, counts: typing.Dict[states.State, typing.Dict[states.State, int]]) -> typing.Dict[states.State, typing.Dict[states.State, float]]: ...
    def add_state(self, state: states.State, inner: bool = ..., outer: bool = ...) -> None: ...
    def remove_state(self, state: states.State, inner: bool = ..., outer: bool = ...) -> None: ...
