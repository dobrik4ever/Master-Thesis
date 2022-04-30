from cellpose import models
from skimage import io
import numpy as np
from tqdm import trange
import os

def cellpose_segmenter(folder_path, stack_fname, gpu):
    path = os.path.join(folder_path, stack_fname)

    print(f'> Loading {stack_fname}')
    stack = io.imread(path)
    depth = stack.shape[0]
    print(f'> Loaded  {stack_fname}, shape: {stack.shape}, {depth:=}, dtype: {stack.dtype}')

    print('> Initializing GPU model...') if gpu else print('Initializing CPU model...')
    model = models.Cellpose(gpu=gpu, model_type='cyto')
    output = np.zeros_like(stack)
    for slice in trange(depth):
        mask, flows, styles, diams = model.eval(stack[slice], channels=[0,0], diameter = None, tile=False, min_size=-1)
        output[slice] = mask

    path_to_save = os.path.join(folder_path, 'masks.npy')
    np.save(path_to_save, output)
    print('> Saved to', path_to_save)
    print('> Done!')
    return mask


# ====== End of parameter section ====== #
if __name__ == '__main__':
    cellpose_segmenter( folder = '4 Raw data',
                        file = 'stack_R.tif',
                        gpu = False)


