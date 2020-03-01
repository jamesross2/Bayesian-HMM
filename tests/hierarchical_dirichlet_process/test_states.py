import numpy
import pytest

import bayesian_hmm


def test_symbol() -> None:
    vals = ["sym", "sym2", 1, -100]
    badvals = [[1, 2], {50, 48}]

    # symbols created correctly
    for val in vals:
        symbol = bayesian_hmm.State(val)
        assert symbol.value == val
        str(symbol)

    # non-symbol values fail
    for badval in badvals:
        with pytest.raises(ValueError, match="Symbols must have hashable"):
            bayesian_hmm.State(badval)

    # Symbols with equal value use a single dictionary entry
    s1 = bayesian_hmm.State(1)
    s2 = bayesian_hmm.State(1)
    assert s1 is not s2
    assert s1 == s2
    assert len({s1, s2}) == 1
    counts = {s1: 10, s2: 20}
    assert counts[s1] == 20


def test_printing() -> None:
    symbol = bayesian_hmm.State(1)
    assert str(symbol) == "1"
    assert repr(symbol) == "(1)"

    symbol = bayesian_hmm.State("Longname")
    assert str(symbol) == "Longname"
    assert repr(symbol) == "(Longname)"


def test_comparisons() -> None:
    """Test symbol comparisons to ensure that hashing is done properly."""
    symbol1 = bayesian_hmm.State(1)
    symbol_high = bayesian_hmm.State(667723)
    symbol_low = bayesian_hmm.State(-numpy.Inf)

    assert symbol_low < symbol1 < symbol_high
