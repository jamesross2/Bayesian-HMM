import bayesian_hmm
import pytest


def test_symbol() -> None:
    vals = ["sym", "sym2", 1, -100]
    badvals = [[1,2], {50, 48}]
    for val in vals:
        symbol = bayesian_hmm.Symbol(val)
        assert symbol.value == val
        str(symbol)
    for badval in badvals:
        with pytest.raises(ValueError, match="Symbols must have hashable"):
            bayesian_hmm.Symbol(badval)
