from src.utilities.penalty_functions.IPenalty import IPenalty

class GeneralizedCharbonnierPenalty(IPenalty):
    """
    f(x)=(x**2 + epsilon**2)**a
    """

    def __init__(self,a=0.45,epsilon=0.001):
        self._a = a
        self._epsilon = epsilon

    def get_value_at(self, x):
        return (x**2 + self._epsilon**2)**self._a

    def get_first_derivative_at(self, x):
        return 2*self._a*x*(x**2 + self._epsilon**2)**(self._a-1)

    def get_second_derivative_at(self, x):
        #x2_e2 = x**2 + self._epsilon**2
        #a = self._a
        #return 2*a*(2*(a-1)*(x**2)*(x2_e2**(a-2))+(x2_e2**(a-1)))
        return 2 * self._a * (x ** 2 + self._epsilon ** 2) ** (self._a - 1)
