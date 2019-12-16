import bayesian_hmm
import numpy


def test_mcmc():
    # create emission sequences
    base_sequence = list(range(5))
    sequences = [base_sequence * 5 for _ in range(5)]

    # initialise object with overestimate of true number of latent states
    hmm = bayesian_hmm.HDPHMM(sequences)
    hmm.initialise(k=20)

    # estimate hyperparameters, making use of multithreading functionality
    results = hmm.mcmc(n=100, burn_in=20)

    # specify expected dict keys at nested levels
    expected_results_keys = [
        "state_count",
        "loglikelihood",
        "chain_loglikelihood",
        "hyperparameters",
        "beta_emission",
        "beta_transition",
        "parameters",
    ]
    expected_hyperparameters_keys = [
        "alpha",
        "gamma",
        "alpha_emission",
        "gamma_emission",
        "kappa",
    ]

    # check that results contains expected elements
    assert len(results) == 7
    assert list(results.keys()) == expected_results_keys
    assert all(len(r) == 8 for r in results.values())
    assert all(type(r) == list for r in results.values())

    # state count and calculate_loglikelihood are straightforward sequences
    assert all(type(x) == int for x in results["state_count"])
    assert all(
        type(x) == numpy.float64
        for k in ["loglikelihood", "chain_loglikelihood"]
        for x in results[k]
    )

    # hyperparameters is a list of dicts with atmoic values
    assert all(type(x) == dict for x in results["hyperparameters"])
    assert all(
        list(x.keys()) == expected_hyperparameters_keys
        for x in results["hyperparameters"]
    )
    assert all(type(y) == float for x in results["hyperparameters"] for y in x.values())
    assert all(y >= 0 for x in results["hyperparameters"] for y in x.values())

    # beta_emission and beta transition are dicts of floats
    assert all(type(x) == dict for x in results["beta_emission"])
    assert all(type(y) == float for x in results["beta_emission"] for y in x.values())
    assert all(type(x) == dict for x in results["beta_transition"])
    assert all(type(y) == float for x in results["beta_transition"] for y in x.values())

    # parameters is a list of dicts, with each dict a point-in-time snap of parameters
    assert all(type(x) == dict for x in results["parameters"])
    assert all(isinstance(y, dict) for x in results["parameters"] for y in x.values())
    assert all(
        isinstance(y, bayesian_hmm.Symbol)
        for x in results["parameters"]
        for y in x["p_initial"]
    )
    assert all(
        type(y) == dict
        for x in results["parameters"]
        for y in x["p_transition"].values()
    )
    assert all(
        type(y) == dict for x in results["parameters"] for y in x["p_emission"].values()
    )

    # TODO: check that internal structure of p_* histories is consistent
