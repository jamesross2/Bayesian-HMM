"""A class to store states and emissions abstractly."""

import collections
import functools
import typing


# Symbol objects are essentially wrappers for any hashable atom
@functools.total_ordering
class State(object):
    ___slots__ = ("value",)

    def __init__(self, value: typing.Union[str, float, int, None]):
        if not isinstance(value, collections.abc.Hashable):
            raise ValueError("Symbols must have hashable (and immutable) values.")
        self.value = value
        self.__special: typing.Optional[str] = None

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"({self.value})"

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            return self.value == other.value and self.__special == other.__special
        else:
            raise NotImplementedError("Cannot compare Symbol object to a {}".format(type(other)))

    def __lt__(self, other) -> bool:
        if isinstance(other, State):
            return self.value < other.value
        else:
            raise NotImplementedError("Cannot compare Symbol object to a {}".format(type(other)))


# Create some special symbols for standalone states
class StartingState(State):
    def __init__(self):
        super(StartingState, self).__init__(None)
        self.__special = "start"

    def __str__(self) -> str:
        return "StartingState"

    def __repr__(self) -> str:
        return f"({self.__special})"


# define a single empty symbol also
class AggregateState(State):
    def __init__(self):
        super(AggregateState, self).__init__(None)
        self.__special = "aggr"

    def __str__(self) -> str:
        return "AggregateState"

    def __repr__(self) -> str:
        return f"({self.__special})"
