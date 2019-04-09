import bayesian_hmm
import numpy


def test_mcmc():
    # create emission sequences
    base_sequence = list(range(5))
    sequences = [base_sequence * 5 for _ in range(5)]

    # initialise object with overestimate of true number of latent states
    hmm = bayesian_hmm.HDPHMM(sequences)
    hmm.initialise(k=20)

    # estimate parameters, making use of multithreading functionality
    results = hmm.mcmc(n=100, burn_in=20)

    # check that results contains expected elements
    assert len(results) == 6
    assert all(len(r) == 8 for r in results)
    assert all(type(r) == list for r in results)
    assert all(len(x) == 5 for x in results[2])

    # check that elements have expected types
    observed_types = list(map(lambda r: type(r[0]), results))
    expected_types = [int, numpy.float64, tuple, dict, dict, tuple]
    assert observed_types == expected_types
    assert all(type(x) == float for hyperparams in results[2] for x in hyperparams)
    assert all(x >= 0 for hyperparams in results[2] for x in hyperparams)
