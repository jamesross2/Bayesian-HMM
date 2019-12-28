import bayesian_hmm
import pytest


def test_bayesian_variable() -> None:
    """Check that parent class works as expected."""
    bayes_var = bayesian_hmm.hierarchical_dirichlet_process.Variable()
    assert isinstance(bayes_var, bayesian_hmm.hierarchical_dirichlet_process.Variable)
    assert bayes_var.value is None

    # check that parent methods return helpful errors
    with pytest.raises(NotImplementedError, match="Bayesian variables must implement a 'likelihood' method."):
        bayes_var.log_likelihood()
    with pytest.raises(NotImplementedError, match="Bayesian variables must define a 'resample' method."):
        bayes_var.resample()