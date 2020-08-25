
class SolverSettings:
    pass

def solve_layer(frist_frame,first_frame_derivative,second_frame,second_frame_derivative, initial_flow_field, solver_settings):
    """

    :param frist_frame: np.array(float) shape = (ColorChannel,Height,Width)
    :param first_frame_derivative: np.array(float) shape = (ColorChannel,Derivative_Direction,Height,Width)
    :param second_frame: np.array(float) shape = (ColorChannel,Height,Width)
    :param second_frame_derivative: np.array(float) shape = (ColorChannel,Derivative_Direction,Height,Width)
    :param initial_flow_field: np.array(float) (Flow_Direction, Height,Width)
    :param solver_settings: SolverSettings
    :return: np.array(float) (Flow_Direction, Height,Width)
    """
    