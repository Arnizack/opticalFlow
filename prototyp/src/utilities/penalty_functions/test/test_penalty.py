from src.utilities.penalty_functions.IPenalty import IPenalty
import numpy as np
import matplotlib.pyplot as plt


def plot_penalty_function(func : IPenalty, start = -2, stop = 2, space = 0.1):
    x = np.arange(start,stop,space)
    y = func.get_value_at(x)
    first_derivative_y = func.get_first_derivative_at(x)
    second_derivative_y = func.get_second_derivative_at(x)

    fig, axs = plt.subplots(1, 3)

    axs[0].plot(x,y)
    axs[0].set_title("Value")

    axs[1].plot(x, first_derivative_y)
    axs[1].set_title("First derivative")

    axs[2].plot(x, second_derivative_y)
    axs[2].set_title("Second derivative")

    plt.show()