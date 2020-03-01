import abc
import typing


class Variable(object, metaclass=abc.ABCMeta):
    """A parent class for Bayesian variables within the Bayesian-HMM package."""

    @abc.abstractmethod
    def __init__(self) -> None:
        # self.value: typing.Union[int, float, None] = None
        pass

    # @abc.abstractmethod
    # def log_likelihood(self, **kwargs) -> float:
    #     raise NotImplementedError("Bayesian variables must implement a 'likelihood' method.")
    #
    # @abc.abstractmethod
    # def resample(self, **kwargs) -> typing.Any:
    #     raise NotImplementedError("Bayesian variables must define a 'resample' method.")
