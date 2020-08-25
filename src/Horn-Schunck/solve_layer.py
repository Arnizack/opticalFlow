
class SolverSettings:
    pass

def solve_layer(frist_frame,first_frame_derivative,second_frame,second_frame_derivative, initial_flow_field, solver_settings):
    """

    :param frist_frame: np.array(float) shape = (ColorChannel,Width,Height)
    :param first_frame_derivative: np.array(float) shape = (ColorChannel,Derivative_Direction,Width,Height)
    :param second_frame: np.array(float) shape = (ColorChannel,Width,Height)
    :param second_frame_derivative: np.array(float) shape = (ColorChannel,Derivative_Direction,Width,Height)
    :param initial_flow_field: np.array(float) (Flow_Direction, Width,Height)
    :param solver_settings: SolverSettings
    :return: np.array(float) (Flow_Direction, Width,Height)
    """
    