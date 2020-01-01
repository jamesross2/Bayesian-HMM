import pytest

import bayesian_hmm


def test_bayesian_variable() -> None:
    """Check that parent class works as expected."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class Variable"):
        _ = bayesian_hmm.hierarchical_dirichlet_process.Variable()
