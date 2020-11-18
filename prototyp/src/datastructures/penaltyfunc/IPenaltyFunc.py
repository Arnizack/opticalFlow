from abc import ABC

class IPenaltyFunc(ABC):
    @abstractmethod
    def value(self,x):
        pass

    @abstractmethod
    def first_derivative(self,x):
        pass

    @abstractmethod
    def second_derivative(self,x):
        pass
