import numpy as np

def scale_np_array_in_range(array : np.ndarray, start = -1, end = 1) -> np.ndarray :
    array-=array.min()
    array= array / array.max()

    array*=(end-start)
    array+=start
    return array

def scale_image_channels_in_range(img:np.ndarray,start = -1, end = 1) -> np.ndarray :
    for idx, channel in enumerate(img):
        img[idx] = scale_np_array_in_range(channel,start=start,end=end)
    return img
