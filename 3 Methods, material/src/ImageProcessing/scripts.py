import numpy as np
from scipy import ndimage

class Denoiser:
    
    def median_filter(image: np.array, size: int):
        return ndimage.median_filter(image, size=[size, size, size])

    def gaussian_filter(image: np.array, sigma: int):
        return ndimage.gaussian_filter(image, sigma=sigma)

    def depth_average(image: np.array, num_slices: int):
        depth = image.shape[0]//num_slices
        new_image = np.zeros([depth, image.shape[1], image.shape[2]])

        for i in range(depth):
            new_image[i] = np.average(image[i*num_slices:i*num_slices + num_slices], axis=0)

        return new_image

    def depth_median(image: np.array, num_slices: int):
        depth = image.shape[0]//num_slices
        new_image = np.zeros([depth, image.shape[1], image.shape[2]])

        for i in range(depth):
            new_image[i] = np.median(image[i*num_slices:i*num_slices + num_slices], axis=0)

        return new_image

    def max_projection(image: np.array):
        return np.max(image, axis=0)