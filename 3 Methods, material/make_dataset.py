from src.utils.general import make_training_set
from src.utils.cellpose_segment import cellpose_segmenter
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

mask, = cellpose_segmenter( folder_path = '4 Raw data',
                    stack_name = 'max_expanded.tif',
                    three_axes = False,
                    cell_diameter = 23,
                    min_size = 300,
                    gpu = False     
                                )
stack = io.imread('4 Raw data/max_expanded.tif')
mask = np.load('4 Raw data/max_expanded_mask_XY.npy')
plt.imshow(stack[0], cmap='gray')
plt.imshow(mask[0], cmap='jet', alpha=0.5)
plt.show()
cells = make_training_set(stack[0], mask[0], offset = 5, make_zero=False, plot=True)
# io.imsave('')