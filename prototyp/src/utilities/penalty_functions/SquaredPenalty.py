from src.utilities.penalty_functions.IPenalty import IPenalty


class SquaredPenalty(IPenalty):
    """
    f(x) = x**2
    """
    def get_value_at(self, x):
        return x ** 2

    def get_first_derivative_at(self, x):
        return 2 * x

    def get_second_derivative_at(self, x):
        return x*0+2
