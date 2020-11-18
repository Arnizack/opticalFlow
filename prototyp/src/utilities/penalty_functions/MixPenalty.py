from src.utilities.penalty_functions.IPenalty import IPenalty


class MixPenalty(IPenalty):
    def __init__(self, first_penalty : IPenalty, second_penalty : IPenalty, mix_factor : float):
        self._first_penalty = first_penalty
        self._second_penalty = second_penalty
        self.mix_factor = mix_factor

    def get_value_at(self, x):
        first = self._first_penalty.get_value_at(x)
        second = self._second_penalty.get_value_at(x)
        return first * (1 - self.mix_factor) + second * self.mix_factor

    def get_first_derivative_at(self, x):
        first = self._first_penalty.get_first_derivative_at(x)
        second = self._second_penalty.get_first_derivative_at(x)
        return first * (1 - self.mix_factor) + second * self.mix_factor

    def get_second_derivative_at(self, x):
        first = self._first_penalty.get_second_derivative_at(x)
        second = self._second_penalty.get_second_derivative_at(x)
        return first * (1 - self.mix_factor) + second * self.mix_factor
