from src.utilities.penalty_functions.SquaredPenalty import SquaredPenalty
from src.utilities.penalty_functions.GeneralizedCharbonnierPenalty import GeneralizedCharbonnierPenalty
from src.utilities.penalty_functions.MixPenalty import MixPenalty

def get_values(blend_factor_list, x_list):

    epsilon = 0.001;
    a = 0.45;
    squared = SquaredPenalty()
    charbonnier = GeneralizedCharbonnierPenalty(a=a,epsilon=epsilon)

    results_value = []
    results_1Deriv = []
    results_2Deriv = []

    for mix_factor in blend_factor_list:
        penalty = MixPenalty(squared,charbonnier,mix_factor)
        for x in x_list:
            results_value.append(penalty.get_value_at(x))
            results_1Deriv.append(penalty.get_first_derivative_at(x))
            results_2Deriv.append(penalty.get_second_derivative_at(x))

    return  (results_value,results_1Deriv,results_2Deriv)

if __name__ == '__main__':
    print(get_values([0,0.5,1],[0,0.2,0.6,1,5]))