import abc

class Variable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self): ...
