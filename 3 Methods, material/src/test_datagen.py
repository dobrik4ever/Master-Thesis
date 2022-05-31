from DataSimulation.DataGenerator import DataGenerator, tensor_to_coordinates

folder = '../data'
dt = DataGenerator(folder, 1)
dt.generate()

from skimage import io
import numpy as np
import os
from matplotlib import pyplot as plt

img = np.load(f'{folder}/img_0.npy')
label = np.load(f'{folder}/pos_0.npy')
coords = tensor_to_coordinates(label, 30, 300)
plt.imshow(img, cmap='gray')
plt.scatter(coords[0], coords[1], s=10, c='r')
plt.show()


