"""A class to store states and emissions abstractly."""

import collections
import functools
import typing


# Symbol objects are essentially wrappers for any hashable atom
@functools.total_ordering
class State(object):
    ___slots__ = ("value",)

    def __init__(self, value: collections.abc.Hashable):
        if not isinstance(value, collections.abc.Hashable):
            raise ValueError("Symbols must have hashable (and immutable) values.")
        self.value: collections.abc.Hashable = value

    @property
    def special(self) -> bool:
        return isinstance(self, SpecialState)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"({self.value})"

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __lt__(self, other) -> bool:
        return self.value < other.value


class SpecialState(State):
    def __init__(self, value: str):
        super(SpecialState, self).__init__(value)

    def __eq__(self, other) -> bool:
        return other.special and self.value == other.value

    # must re-implement hash for subclass that also defines eq
    def __hash__(self) -> int:
        return hash(self.value)


# Create some special symbols for standalone states
class StartingState(SpecialState):
    def __init__(self):
        super(StartingState, self).__init__("start")


# define a single empty symbol also
class AggregateState(SpecialState):
    def __init__(self):
        super(AggregateState, self).__init__("aggregate")


# a class to store 'missing' observations
class MissingState(SpecialState):
    def __init__(self):
        super(MissingState, self).__init__("missing")
