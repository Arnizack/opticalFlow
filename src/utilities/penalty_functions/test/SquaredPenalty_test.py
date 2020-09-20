from src.utilities.penalty_functions.SquaredPenalty import SquaredPenalty
from src.utilities.penalty_functions.test.test_penalty import plot_penalty_function

if __name__ == '__main__':
    func = SquaredPenalty()
    plot_penalty_function(func)
