import numpy
import scipy.stats

import bayesian_hmm


# TODO: expand tests
def test_stick_breaking_process() -> None:
    # create a standard variable
    alpha = bayesian_hmm.hierarchical_dirichlet_process.Hyperparameter(prior=lambda: 0, log_likelihood=lambda x: 0)
    stick_breaking_process = bayesian_hmm.hierarchical_dirichlet_process.StickBreakingProcess(alpha=alpha)

    # check that initialisation follows correctly
    assert isinstance(stick_breaking_process, bayesian_hmm.StickBreakingProcess)
    assert isinstance(stick_breaking_process.alpha, bayesian_hmm.Hyperparameter)
    assert alpha is stick_breaking_process.alpha
    assert stick_breaking_process.value == {bayesian_hmm.EmptySymbol(): 1}


def test_log_likelihood() -> None:
    # create a standard variable
    alpha = bayesian_hmm.hierarchical_dirichlet_process.Hyperparameter(prior=lambda: 1.0, log_likelihood=lambda x: 0.0)
    stick_breaking_process = bayesian_hmm.hierarchical_dirichlet_process.StickBreakingProcess(alpha=alpha)

    # likelihood for empty variable is easy
    assert stick_breaking_process.log_likelihood() == 0.0

    # when alpha is one (see above), beta variables have uniform distribution, so log likelihood is still zero
    stick_breaking_process.add_state(bayesian_hmm.Symbol("A"))
    stick_breaking_process.add_state(bayesian_hmm.Symbol("B"))
    assert stick_breaking_process.log_likelihood() == 0.0

    # when alpha grows, large betas are unlikely
    stick_breaking_process.alpha.value = 3
    stick_breaking_process.value[bayesian_hmm.Symbol("A")] = 0.8
    stick_breaking_process.value[bayesian_hmm.Symbol("B")] = 0.15
    stick_breaking_process.value[bayesian_hmm.EmptySymbol()] = 0.05
    assert stick_breaking_process.log_likelihood() < 0.0
    assert numpy.isclose(
        stick_breaking_process.log_likelihood(),
        scipy.stats.beta.logpdf(x=0.8, a=1, b=3) + scipy.stats.beta.logpdf(x=0.75, a=1, b=3),
    )

    # change values of beta so that likelihood is fixed
