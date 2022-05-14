from cellpose import models
from docutils import DataError
from skimage import io
import numpy as np
from tqdm import trange
import os

# TODO: make it more class based and easier to use. Must not open stack, must take stack as input for example. Not extensible code
def cellpose_segmenter(folder_path, stack_name, gpu = False, three_axes = False, cell_diameter = None, min_size:int =-1):
    """Script to generate masks from a stack using Cellpose.

    Args:
        folder_path (str): path to a folder, where the stack is stored
        stack_name (str): stack file name
        gpu (bool, optional): Flag that indicates GPU computation. Defaults to False.
        three_axes (bool, optional): Flag to select either only XY segmentation or XY, XZ, YZ. Defaults to False.

    Returns:
        _type_: _description_
    """
    path = os.path.join(folder_path, stack_name)
    print('> Segmentation will be performed on:', path, end=' ')
    if three_axes: print(' (XY, XZ, YZ)')
    else: print(' (XY)')

    print(f'> Loading {stack_name}')
    stack = io.imread(path)
    if len(stack.shape) != 3: raise DataError(f'Stack must be of shape Z x H x W, got instead {stack.shape}')
    depth, height, width = stack.shape
    print(f'> Loaded  {stack_name}, shape: {stack.shape}, dtype: {stack.dtype}')
    stack_name = stack_name.split('.')[0]
    print('> Initializing GPU model...') if gpu else print('Initializing CPU model...')
    model = models.Cellpose(gpu=gpu, model_type='cyto')

    output_XY = np.zeros_like(stack)
    for slice in trange(depth):
        mask, flows, styles, diams = model.eval(stack[slice], channels=[0,0], diameter = cell_diameter, tile=False, min_size=min_size)
        output_XY[slice] = mask
    path_to_save = os.path.join(folder_path, f'{stack_name}_mask_XY.npy')
    np.save(path_to_save, output_XY)
    print('> Saved to', path_to_save)

    if three_axes:

        output_XZ = np.zeros_like(stack)
        for slice in trange(height):
            mask, flows, styles, diams = model.eval(stack[:, slice], channels=[0,0], diameter = cell_diameter, tile=False, min_size=min_size)
            output_XZ[:, slice] = mask
        path_to_save = os.path.join(folder_path, f'{stack_name}_mask_XZ.npy')
        np.save(path_to_save, output_XZ)
        print('> Saved to', path_to_save)

        output_YZ = np.zeros_like(stack)
        for slice in trange(width):
            mask, flows, styles, diams = model.eval(stack[:, :, slice], channels=[0,0], diameter = cell_diameter, tile=False, min_size=min_size)
            output_YZ[:, :, slice] = mask
        path_to_save = os.path.join(folder_path, f'{stack_name}_mask_YZ.npy')
        np.save(path_to_save, output_YZ)
        print('> Saved to', path_to_save)
    
    print('> Done!')
    if not three_axes:
        return (output_XY,)
    return (output_XY, output_XZ, output_YZ)


# ====== End of parameter section ====== #
if __name__ == '__main__':
    cellpose_segmenter( folder_path = '4 Raw data',
                        stack_name = 'max.tif',
                        three_axes = False,
                        gpu = False                 )


