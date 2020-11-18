from src.utilities.image_derivative import differentiate_images
from src.utilities.image_pyramid import create_image_pyramid

class ImagePair:
    """
    Saves 2 images and it's derivatives in different scales
    """
    #Shape (PyramidsLevels,ColorChannels,Width,Height)
    first_image_pyramid = None
    # Shape (PyramidsLevels,ColorChannels,2,Width,Height)
    first_image_derivative_pyramid = None

    # Shape (PyramidsLevels,ColorChannels,Width,Height)
    second_image_pyramid = None
    # Shape (PyramidsLevels,ColorChannels,2,Width,Height)
    second_image_derivative_pyramid = None

    def __init__(self,first_image,second_image,scalefactors):
        self.first_image_pyramid=create_image_pyramid(first_image,scalefactors)
        self.first_image_pyramid = differentiate_images(self.first_image_pyramid)

        self.second_image_pyramid = create_image_pyramid(second_image,scalefactors)
        self.second_image_derivative_pyramid = differentiate_images(self.second_image_pyramid)



