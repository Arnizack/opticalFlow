from abc import ABC, abstractmethod


class IPenalty(ABC):
    @abstractmethod
    def get_value_at(self,x):
        pass

    @abstractmethod
    def get_first_derivative_at(self, x):
        pass

    @abstractmethod
    def get_second_derivative_at(self, x):
        pass
