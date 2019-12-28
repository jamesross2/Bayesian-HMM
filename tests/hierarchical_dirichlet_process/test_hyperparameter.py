import bayesian_hmm


def test_hyperparameter() -> None:
    """Check that hyperparameters follow Bayesian rules."""
    # create degenerate hyperparameter to test that functions work
    bayes_hyperparameter = bayesian_hmm.hierarchical_dirichlet_process.Hyperparameter(
        prior=lambda: 3,
        log_likelihood=lambda x: 0
    )

    # initialised correctly
    assert bayes_hyperparameter.value == 3
    assert bayes_hyperparameter.log_likelihood() == 0

    # functions work
    bayes_hyperparameter.resample(force=True)
    assert bayes_hyperparameter.value == 3
    bayes_hyperparameter.resample(force=False)
    assert bayes_hyperparameter.value == 3
