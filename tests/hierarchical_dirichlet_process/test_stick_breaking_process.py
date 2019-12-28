import bayesian_hmm


# TODO: expand tests
def test_stick_breaking_process() -> None:
    # create a standard variable
    alpha = bayesian_hmm.hierarchical_dirichlet_process.Hyperparameter(prior=lambda: 0, log_likelihood=lambda x: 0)
    # stick_breaking_process = bayesian_hmm.hierarchical_dirichlet_process.StickBreakingProcess(alpha=alpha)

    # initialised correctly?
    # assert stick_breaking_process.value == []
