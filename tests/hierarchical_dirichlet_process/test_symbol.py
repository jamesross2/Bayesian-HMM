import bayesian_hmm
import pytest


def test_symbol() -> None:
    vals = ["sym", "sym2", 1, -100]
    badvals = [[1, 2], {50, 48}]

    # symbols created correctly
    for val in vals:
        symbol = bayesian_hmm.Symbol(val)
        assert symbol.value == val
        str(symbol)

    # non-symbol values fail
    for badval in badvals:
        with pytest.raises(ValueError, match="Symbols must have hashable"):
            bayesian_hmm.Symbol(badval)

    # Symbols with equal value use a single dictionary entry
    s1 = bayesian_hmm.Symbol(1)
    s2 = bayesian_hmm.Symbol(1)
    assert s1 is not s2
    assert s1 == s2
    assert len({s1, s2}) == 1
    counts = {s1: 10, s2: 20}
    assert counts[s1] == 20
