from src.utilities.penalty_functions.GeneralizedCharbonnierPenalty import GeneralizedCharbonnierPenalty
from src.utilities.penalty_functions.test.test_penalty import plot_penalty_function

if __name__ == '__main__':
    func = GeneralizedCharbonnierPenalty(epsilon=0.01)
    plot_penalty_function(func,start=-0.5,stop = 0.5,space=0.001)