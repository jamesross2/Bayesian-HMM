import typing


class Variable(object):
    """A parent class for Bayesian variables within the Bayesian-HMM package."""

    def __init__(self) -> None:
        self.value: typing.Union[int, float, None] = None
        pass

    def log_likelihood(self, *args):
        raise NotImplementedError(
            "Bayesian variables must implement a 'likelihood' method."
        )

    def resample(self, *args):
        raise NotImplementedError("Bayesian variables must define a 'resample' method.")
