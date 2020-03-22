import warnings

import pytest

import bayesian_hmm


# use arithmetic series function to force incrementing likelihood/parameter values
class HyperFunctions(object):
    def __init__(self, delta=1):
        self.val = 0
        self.delta = delta

    def newval(self):
        self.val += self.delta
        return self.val


def test_hyperparameter() -> None:
    """Check that hyperparameters follow Bayesian rules."""
    # create degenerate hyperparameter to test that functions work
    bayes_hyperparameter = bayesian_hmm.hyperparameter.Hyperparameter(prior=lambda: 3, log_likelihood=lambda x: 0)

    # initialised correctly
    assert bayes_hyperparameter.value == 3
    assert bayes_hyperparameter.log_likelihood() == 0

    # functions work
    bayes_hyperparameter.resample(force=True)
    assert bayes_hyperparameter.value == 3
    bayes_hyperparameter.resample(force=False, posterior_log_likelihood=lambda x: 0.0)
    assert bayes_hyperparameter.value == 3


def test_resample() -> None:
    # use parameter that should increase by 1 each resample
    hyper_functions = HyperFunctions()
    bayes_hyperparameter = bayesian_hmm.hyperparameter.Hyperparameter(
        prior=hyper_functions.newval, log_likelihood=lambda x: 0.0
    )

    # check that rejected if needed
    with pytest.raises(ValueError, match="Posterior likelihood required if force is False."):
        bayes_hyperparameter.resample()

    # check that parameter increases by arithmetic series
    starting_value = bayes_hyperparameter.value
    output = bayes_hyperparameter.resample(force=False, posterior_log_likelihood=lambda x: 0.0)
    assert bayes_hyperparameter.value == output
    assert output == starting_value + 1

    # check that parameter is accepted when likelihood grows
    num_resamples = 100
    hyper_ll = HyperFunctions()
    starting_val = bayes_hyperparameter.value
    outputs = [
        bayes_hyperparameter.resample(posterior_log_likelihood=lambda x: hyper_ll.newval())
        for _ in range(num_resamples)
    ]
    assert outputs == list(range(starting_val + 1, num_resamples + starting_value + 2))

    # warning based test if unlikely parameter accepted (technically possible so do not raise error)
    hyper_ll = HyperFunctions(delta=-10)
    outputs = [
        bayes_hyperparameter.resample(posterior_log_likelihood=lambda x: hyper_ll.newval())
        for _ in range(num_resamples)
    ]
    if len(set(outputs)) > 50:
        warnings.warn("Check unlikely result: many hyperparameters with low likelihood accepted.")
