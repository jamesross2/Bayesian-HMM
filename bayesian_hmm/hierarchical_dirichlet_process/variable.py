from abc import ABCMeta, abstractmethod


class Variable(object, metaclass=ABCMeta):
    """A parent class for Bayesian variables within the Bayesian-HMM package."""

    @abstractmethod
    def __init__(self) -> None:
        # self.value: typing.Union[int, float, None] = None
        pass

    @abstractmethod
    def log_likelihood(self, *args):
        raise NotImplementedError("Bayesian variables must implement a 'likelihood' method.")

    @abstractmethod
    def resample(self, *args):
        raise NotImplementedError("Bayesian variables must define a 'resample' method.")
