from src.datastructures.penaltyfunc import IPenaltyFunc

class QuadraticPenaltyFunc(IPenaltyFunc):
    def value(self, x):
        return x**2

    def first_derivative(self, x):
        return 2*x

    def second_derivative(self, x):
        return 2