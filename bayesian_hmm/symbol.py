#!/usr/bin/env python3
"""A class to store states and emissions abstractly."""

import collections
import typing
import functools


# Symbol objects are essentially wrappers for any hashable atom
@functools.total_ordering
class Symbol(object):
    ___slots__ = ("value",)

    def __init__(self, value: typing.Union[str, float, int, None]):
        if not isinstance(value, collections.Hashable):
            raise ValueError("Symbols must have hashable (and immutable) values.")
        self.value = value

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return "({})".format(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, Symbol):
            return self.value == other.value
        else:
            raise NotImplementedError(
                "Cannot compare Symbol object to a {}".format(type(other))
            )

    def __lt__(self, other) -> bool:
        if isinstance(other, Symbol):
            return self.value < other.value
        else:
            raise NotImplementedError(
                "Cannot compare Symbol object to a {}".format(type(other))
            )


# define a single empty symbol also
def EmptySymbol() -> Symbol:
    return Symbol(None)
