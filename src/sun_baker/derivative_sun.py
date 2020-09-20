from src.utilities.image_derivative import differentiate_matrix

def derivative_sun(gray_first_image,gray_second_image):
    #See: A Quantitative Analysis of Current Practices in Optical Flow Estimation
    #and the Principles behind Them
    first_I_y,first_I_x = differentiate_matrix(gray_first_image)
    second_I_y, second_I_x = differentiate_matrix(gray_second_image)
    b=0.4

    I_x = first_I_x * b + second_I_x * (1-b)
    I_y = first_I_y * b + second_I_y * (1-b)
    I_t = gray_second_image-gray_first_image


    return I_x,I_y,I_t