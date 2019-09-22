#!/usr/bin/env python3
"""
Bayesian Hierarchical Dirichlet process class.
"""
# Support typehinting.
from __future__ import annotations
from typing import Any, Union, Optional, Set, Dict, Iterable, List, Callable, Generator

# Common typing aliases
Numeric = Union[int, float]
DictStrNum = Dict[str, Numeric]
DictStrDictStrNum = Dict[str, DictStrNum]


# a Bayesian
class HierarchicalDirichlet(object):
    def __init__(self) -> None:
        # create object to store counts
        self.n_transition: DictStrDictStrNum
        
    
    
