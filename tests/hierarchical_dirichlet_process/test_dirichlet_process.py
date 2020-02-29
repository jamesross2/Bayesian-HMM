import typing

import numpy

import bayesian_hmm


def create_dirichlet_process() -> bayesian_hmm.DirichletProcessFamily:
    alpha = bayesian_hmm.hyperparameter.Hyperparameter(prior=lambda: 0.4, log_likelihood=lambda x: 0)
    beta = bayesian_hmm.StickBreakingProcess(alpha=alpha)
    gamma = bayesian_hmm.hyperparameter.Hyperparameter(prior=lambda: 3.0, log_likelihood=lambda x: 0)
    kappa = bayesian_hmm.hyperparameter.Hyperparameter(prior=lambda: 0.1, log_likelihood=lambda x: 0)
    dirichlet_process = bayesian_hmm.DirichletProcessFamily(beta=beta, gamma=gamma, kappa=kappa)
    return dirichlet_process


def add_states(dirichlet_process: bayesian_hmm.DirichletProcessFamily, labels: typing.Iterable):
    symbols = {bayesian_hmm.State(x) for x in labels}
    counts = {s1: {s2: int(s1 is not None and s2 is not None) for s2 in symbols} for s1 in symbols}

    # add symbols to process
    for symbol in symbols:
        dirichlet_process.beta.add_state(symbol)

    # information required for further testing
    return set(symbols), counts


def test_initialisation() -> None:
    """Check that auxiliary variables are created correctly."""
    dirichlet_process = create_dirichlet_process()

    # initialised correctly
    assert dirichlet_process.value == {bayesian_hmm.AggregateState(): {bayesian_hmm.AggregateState(): 1}}
    assert isinstance(dirichlet_process.beta, bayesian_hmm.StickBreakingProcess)
    assert isinstance(dirichlet_process.gamma, bayesian_hmm.hyperparameter.Hyperparameter)
    assert isinstance(dirichlet_process.kappa, bayesian_hmm.hyperparameter.Hyperparameter)


def test_stickiness() -> None:
    dirichlet_process_sticky = create_dirichlet_process()
    dirichlet_process_slippery = bayesian_hmm.DirichletProcessFamily(
        beta=dirichlet_process_sticky.beta, gamma=dirichlet_process_sticky.gamma
    )

    # check that stickiness works
    assert dirichlet_process_sticky.sticky
    assert not dirichlet_process_slippery.sticky
    assert isinstance(dirichlet_process_sticky.kappa, bayesian_hmm.hyperparameter.Hyperparameter)
    assert dirichlet_process_slippery.kappa is None


def test_posterior_parameters() -> None:
    # set up
    dirichlet_process = create_dirichlet_process()
    counts = {bayesian_hmm.AggregateState(): {bayesian_hmm.AggregateState(): 0}}
    params = {
        bayesian_hmm.AggregateState(): {bayesian_hmm.AggregateState(): 3.0}
    }  # pulled from degenerate prior for gamma

    # empty posterior parameters should be easy
    assert dirichlet_process.posterior_parameters(counts=counts) == params

    # add symbols for further testing
    symbols, counts = add_states(dirichlet_process, ("a", "b", "c"))

    # check that resampling succeeds
    dirichlet_process.resample(counts=counts)
    assert isinstance(dirichlet_process.value, dict)
    for symbol in symbols:
        assert isinstance(dirichlet_process.value.get(symbol), dict)

    # check that resampling again updates states
    symbols.remove(bayesian_hmm.State("a"))
    dirichlet_process.resample(counts=counts)
    assert isinstance(dirichlet_process.value, dict)
    assert len(dirichlet_process.value) == 3
    for symbol in symbols:
        assert isinstance(dirichlet_process.value.get(symbol), dict)
        assert len(dirichlet_process.value.get(symbol)) == 4


def test_log_likelihood() -> None:
    # set up
    dirichlet_process = create_dirichlet_process()

    # default loglikelihood should be easy
    assert dirichlet_process.log_likelihood() == 0

    # add symbols for further testing (lots of symbols to ensure low probability
    symbols, counts = add_states(dirichlet_process, range(10))
    dirichlet_process.resample(counts=counts)

    # check that log likelihood has updated
    assert dirichlet_process.log_likelihood() != 0


def test_resample() -> None:
    # set up
    dirichlet_process = create_dirichlet_process()
    symbols, counts = add_states(dirichlet_process, range(5))

    # check that resampling succeeds
    dirichlet_process.resample(counts=counts)
    assert isinstance(dirichlet_process.value, dict)
    for symbol in symbols:
        assert isinstance(dirichlet_process.value.get(symbol), dict)
        assert abs(sum(dirichlet_process.value.get(symbol).values()) - 1) < 1e-8

    # check that resampling again updates states
    num_states_curr = len(dirichlet_process.value)
    dirichlet_process.remove_state(bayesian_hmm.State(2))
    dirichlet_process.remove_state(bayesian_hmm.State(3))
    assert isinstance(dirichlet_process.value, dict)
    assert len(dirichlet_process.value) == num_states_curr - 2
    for symbol in dirichlet_process.states_inner:
        assert isinstance(dirichlet_process.value.get(symbol), dict)
        assert len(dirichlet_process.value.get(symbol)) == num_states_curr - 1
        assert abs(sum(dirichlet_process.value.get(symbol).values()) - 1) < 1e-8


def test_add_states() -> None:
    # set up
    dirichlet_process = create_dirichlet_process()
    symbols = {bayesian_hmm.State(x) for x in "syd"}

    # test that symbols are added correctly
    num_symbols = 1
    for symbol in symbols:
        dirichlet_process.beta.add_state(symbol)
        dirichlet_process.add_state(symbol)
        num_symbols += 1
        assert numpy.isclose(sum(dirichlet_process.beta.value.values()), 1)
        assert numpy.isclose(sum(dirichlet_process.value[symbol].values()), 1)
        assert len(dirichlet_process.beta.value.keys()) == num_symbols
