import itertools

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


def test_special_stats() -> None:
    symbol_basic = bayesian_hmm.State(1)
    symbol_start = bayesian_hmm.StartingState()
    symbol_aggr = bayesian_hmm.AggregateState()
    symbol_miss = bayesian_hmm.MissingState()
    symbols = (symbol_basic, symbol_start, symbol_aggr, symbol_miss)

    # special flag works
    assert not symbol_basic.special
    for symbol in (symbol_start, symbol_aggr, symbol_miss):
        assert symbol.special

    # no states are equal
    for state1, state2 in itertools.combinations(symbols, 2):
        assert state1 != state2

    # all states hashable
    for state in symbols:
        assert isinstance(hash(state), int)
