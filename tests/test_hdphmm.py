import numpy
import pytest

import bayesian_hmm


def test_empty_hmm():
    emission_sequences = []
    hmm = bayesian_hmm.HDPHMM(emission_sequences)
    assert not hmm.initialised
    hmm.initialise(0)
    assert hmm.initialised
    assert hmm.emissions == set()
    assert hmm.c == 0
    assert hmm.k == 0
    assert hmm.n == 0


def test_print():
    # checks that printing does not cause an error
    sequences = [["e" + str(x) for x in range(l)] for l in range(1, 50)]
    emission_sequences = [[bayesian_hmm.State(label) for label in sequence] for sequence in sequences]
    hmm = bayesian_hmm.HDPHMM(emission_sequences)
    hmm.initialise(20)
    print(hmm)
    repr(hmm)
    hmm.print_probabilities()
    assert True


def test_initialise_hmm():
    sequences = [["e" + str(x) for x in range(l)] for l in range(1, 50)]
    emission_sequences = [[bayesian_hmm.State(label) for label in sequence] for sequence in sequences]
    hmm = bayesian_hmm.HDPHMM(emission_sequences)
    hmm.initialise(20)

    # check chain initialised correctly
    assert hmm.emissions.symmetric_difference(set(e for seq in emission_sequences for e in seq)) == set()
    assert hmm.c == 49
    assert hmm.k == 20
    assert hmm.n == 49


def test_sticky_initialisation():
    sequences = [[1, 2, 3] * 3] * 3
    emission_sequences = [[bayesian_hmm.State(label) for label in sequence] for sequence in sequences]
    hmm_sticky = bayesian_hmm.HDPHMM(emission_sequences, sticky=True)
    hmm_slippery = bayesian_hmm.HDPHMM(emission_sequences, sticky=False)
    hmm_sticky.initialise(20)
    hmm_slippery.initialise(20)

    # check chain initialises correctly in both cases
    assert 0 <= hmm_sticky.transition_model.kappa.value <= 1
    assert hmm_sticky.transition_model.kappa.prior != (lambda: 0)
    assert hmm_slippery.transition_model.kappa is None


def test_manual_priors():
    sequences = [[1, 2, 3] * 3] * 3
    emission_sequences = [[bayesian_hmm.State(label) for label in sequence] for sequence in sequences]
    priors_default = {
        "alpha": lambda: numpy.random.gamma(2, 2),
        "gamma": lambda: numpy.random.gamma(3, 3),
        "kappa": lambda: numpy.random.beta(1, 1),
        "beta_emission": lambda: numpy.random.gamma(2, 2),
    }
    hmms = {
        "default": bayesian_hmm.HDPHMM(emission_sequences),
        "single": bayesian_hmm.HDPHMM(emission_sequences, priors={"alpha": lambda: -1}),
        "all": bayesian_hmm.HDPHMM(emission_sequences, priors={param: lambda: -1 for param in priors_default.keys()}),
    }

    # check that priors work in all cases
    assert hmms["default"].transition_model.alpha.value > 0
    assert hmms["default"].transition_model.gamma.value > 0
    assert hmms["default"].transition_model.kappa.value > 0
    assert hmms["default"].emission_model.beta.value > 0

    assert hmms["all"].transition_model.alpha.value < 0
    assert hmms["all"].transition_model.gamma.value < 0
    assert hmms["all"].transition_model.kappa.value < 0
    assert hmms["all"].emission_model.beta.value < 0

    assert hmms["single"].transition_model.alpha.value < 0
    assert hmms["single"].transition_model.gamma.value > 0
    assert hmms["single"].transition_model.kappa.value > 0
    assert hmms["single"].emission_model.beta.value > 0

    with pytest.raises(ValueError, match="`sticky` is False, but kappa prior or likelihood function given"):
        _ = bayesian_hmm.HDPHMM(emission_sequences, priors={"kappa": lambda: 2}, sticky=False)
