from src.utilities.warp_grid import warp_matrix
import numpy as np

if __name__ == '__main__':
    data = np.array(
        [[0,1,2,3,4],
         [10,11,12,13,14],
         [20,21,22,23,24],
         [30,31,32,33,34]])

    offset = np.array(
        [
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ]
    )
    actual_result = warp_matrix(data,offset)
    actual_result=actual_result.astype(int)
    expected_result =np.array(
        [[0, 1, 2, 3, 4],
         [10, 12, 12, 13, 14],
         [20, 21, 22, 23, 24],
         [30, 31, 32, 33, 34]])
    print("Functions correctly: ",actual_result==expected_result)
    print(actual_result)