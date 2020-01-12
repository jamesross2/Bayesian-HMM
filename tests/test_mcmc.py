import warnings

import numpy

import bayesian_hmm


def test_mcmc():
    # create emission sequences
    base_labels = list(range(5))
    base_sequence = [bayesian_hmm.State(label) for label in base_labels]
    sequences = [base_sequence * 5 for _ in range(5)]

    # initialise object with overestimate of true number of latent states
    hmm = bayesian_hmm.HDPHMM(sequences)
    hmm.initialise(k=20)

    # estimate hyperparameters, making use of multithreading functionality
    results = hmm.mcmc(n=20, burn_in=4, save_every=4)

    # specify expected dict keys at nested levels
    expected_results_keys = [
        "state_count",
        "loglikelihood",
        "chain_loglikelihood",
        "hyperparameters",
        "beta_emission",
        "beta_transition",
        "emission_probabilities",
        "transition_probabilities",
    ]

    # check that results contains expected elements
    assert len(results) == 8
    assert list(results.keys()) == expected_results_keys
    assert all(len(r) == 4 for r in results.values())
    assert all(type(r) == list for r in results.values())

    # state count and calculate_loglikelihood are straightforward sequences
    assert all(type(x) == int for x in results["state_count"])
    assert all(type(x) == numpy.float64 for k in ["loglikelihood", "chain_loglikelihood"] for x in results[k])

    # hyperparameters is a list of dicts with atomic values
    assert all(type(x) == tuple for x in results["hyperparameters"])
    assert all(type(y) == numpy.float64 for x in results["hyperparameters"] for y in x)
    assert all(y >= 0 for x in results["hyperparameters"] for y in x)

    # betas_are vectors
    beta_history = results["beta_emission"]
    assert all(type(x) == numpy.float64 for x in beta_history)
    assert all(x > 0 for x in beta_history)

    beta_history = results["beta_transition"]
    assert all(type(x) == dict for x in beta_history)
    assert all(type(y) == numpy.float64 for x in beta_history for y in x.values())

    # beta_emission and beta transition are dicts of floats
    for transitions in ("emission_probabilities", "transition_probabilities"):
        transition_history = results[transitions]
        assert all(type(x) == dict for x in transition_history)
        assert all(type(y) == dict for x in transition_history for y in x.values())
        assert all(type(z) == numpy.float64 for x in transition_history for y in x.values() for z in y.values())
        assert all(z >= 0 for x in transition_history for y in x.values() for z in y.values())
