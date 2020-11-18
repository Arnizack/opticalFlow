from src.utilities.angular_error import *
import numpy as np

def normalize_flow_field_test():
    flow = np.array(
        [
            [
                [3,1]
            ],
            [
                [0, 1]
            ],
        ]
    )

    expected_flow = np.array(
        [
            [
                [1,1/np.sqrt(2)]
            ],
            [
                [0,1/np.sqrt(2)]
            ]
        ]
    )

    actual_flow = normalize_flow(flow)

    print(actual_flow==expected_flow)
    print("actual flow")
    print(actual_flow)

def angular_error_test():
    flow1 = np.array(
        [
            [
                [3, 1]
            ],
            [
                [0, 1]
            ],
        ]
    )

    flow2 = np.array(
        [
            [
                [3, 5]
            ],
            [
                [0, -5]
            ],
        ]
    )

    expected_angle = np.array(

        [
            [0, 90]
        ]

    )
    expected_radians=np.radians(expected_angle)
    error = angular_error(flow1,flow2)

    print(expected_radians==error)
    print("Angular error")
    print(error)


if __name__ == '__main__':
    angular_error_test()
