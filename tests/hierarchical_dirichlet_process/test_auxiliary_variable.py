import warnings

import numpy
import pytest
import scipy.stats

import bayesian_hmm
import bayesian_hmm.bayesian_model


def create_auxiliary_var(prior=lambda: 3, log_likelihood=lambda x: 0) -> bayesian_hmm.bayesian_model.AuxiliaryVariable:
    alpha = bayesian_hmm.hyperparameter.Hyperparameter(prior=prior, log_likelihood=log_likelihood)
    beta = bayesian_hmm.StickBreakingProcess(alpha=alpha)
    auxiliary_variable = bayesian_hmm.bayesian_model.AuxiliaryVariable(alpha=alpha, beta=beta)
    return auxiliary_variable


def add_states(auxiliary_variable, labels):
    # add states to variable
    symbols = {bayesian_hmm.State(x) for x in labels}
    for symbol in symbols:
        auxiliary_variable.beta.add_state(symbol)
    counts = {s1: {s2: 1 for s2 in symbols} for s1 in symbols}
    return symbols, counts


def test_initialisation() -> None:
    """Check that auxiliary variables are created correctly."""
    auxiliary_variable = create_auxiliary_var()

    # initialised correctly
    assert auxiliary_variable.value == {}
    assert isinstance(auxiliary_variable.alpha, bayesian_hmm.hyperparameter.Hyperparameter)
    assert isinstance(auxiliary_variable.beta, bayesian_hmm.StickBreakingProcess)


def test_single_log_likelihood() -> None:
    scale = 0.5
    count = 10
    vals_exact = [
        bayesian_hmm.bayesian_model.AuxiliaryVariable.single_variable_log_likelihood(
            scale=scale, value=x, count=count, exact=True
        )
        for x in range(1, count + 1)
    ]
    vals_approx = [
        bayesian_hmm.bayesian_model.AuxiliaryVariable.single_variable_log_likelihood(
            scale=scale, value=x, count=count, exact=False
        )
        for x in range(1, count + 1)
    ]

    # exact values sum close to 1 (complete probability)
    assert abs(sum(numpy.exp(vals_exact)) - 1) < 1e-8

    # approximation within 50% of original
    # TODO: look for ways to improve this approximation
    assert max(abs(numpy.log(x / y)) for x, y in zip(vals_exact, vals_approx)) < 2

    # rejects bad input
    assert (
        bayesian_hmm.bayesian_model.AuxiliaryVariable.single_variable_resample(scale=scale, count=0, exact=False) == 1
    )


def test_single_variable_resample() -> None:
    sample_count = 100
    scale = 1
    count = 20

    # test exact and approximation separately
    for exact in (True, False):
        tests = tuple(
            bayesian_hmm.bayesian_model.AuxiliaryVariable.single_variable_resample(
                scale=scale, count=count, exact=exact
            )
            for _ in range(sample_count)
        )

        # check that results are in correct range
        assert min(tests) >= 0
        assert max(tests) <= count

        if len(set(tests)) < count / 5:
            warnings.warn(
                f"Small number of unique values for auxiliary variables; check unlikely result for exact={exact}."
            )

    # force overflow error with high count
    _ = [
        bayesian_hmm.bayesian_model.AuxiliaryVariable.single_variable_resample(scale=s, count=n, exact=True)
        for s in (0.01, 0.1, 1, 10, 100)
        for n in (1, 10, 100, 1000)
    ]


def test_log_likelihood() -> None:
    # set up
    auxiliary_variable = create_auxiliary_var(prior=lambda: scipy.stats.gamma.rvs(1, 1))

    # default log likelihood should be easy
    with pytest.warns(UserWarning, match="Calculated likelihood of auxiliary variables should not contribute"):
        assert numpy.isclose(auxiliary_variable.log_likelihood(counts=dict()), 0)

    # add states to variable
    symbols, counts = add_states(auxiliary_variable, range(5))
    auxiliary_variable.resample(counts=counts)

    # check that log likelihood has decreased
    with pytest.warns(UserWarning, match="Calculated likelihood of auxiliary variables should not contribute"):
        assert abs(auxiliary_variable.log_likelihood(counts=counts)) < 1e-8


def test_resample() -> None:
    # set up
    auxiliary_variable = create_auxiliary_var(prior=lambda: scipy.stats.gamma.rvs(3, 3))
    symbols, counts = add_states(auxiliary_variable, {"a", "b", "c", "d"})

    # now resample
    auxiliary_variable.resample(counts=counts)
    assert isinstance(auxiliary_variable.value, dict)
    for symbol in symbols:
        assert isinstance(auxiliary_variable.value.get(symbol), dict)
    for key in auxiliary_variable.value.keys():
        assert key in symbols

    # check that resampling again updates states
    symbols.remove(bayesian_hmm.State("c"))
    auxiliary_variable.resample(counts=counts)
    assert isinstance(auxiliary_variable.value, dict)
    assert len(auxiliary_variable.value) == 4
    for symbol in symbols:
        assert isinstance(auxiliary_variable.value.get(symbol), dict)
        assert len(auxiliary_variable.value.get(symbol)) == 4


def test_value_aggregated() -> None:
    # set up
    auxiliary_variable = create_auxiliary_var(prior=lambda: scipy.stats.gamma.rvs(1, 1))
    symbols, counts = add_states(auxiliary_variable, range(5))
    auxiliary_variable.resample(counts=counts)

    # check that aggregated values return as expected
    aggregate_parameters = auxiliary_variable.value_aggregated()
    assert isinstance(aggregate_parameters, dict)
    assert len(aggregate_parameters) == len(symbols)
