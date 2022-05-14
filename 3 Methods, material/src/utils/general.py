import numpy as np
from matplotlib import pyplot as plt
from skimage import io, measure

def draw_contours(mask):
    """Draws contours of a binary mask

    Args:
        mask (np.array): mask array 2D
    """
    contours = measure.find_contours(mask != 0)
    for contour in contours:
        line = plt.plot(contour[:,1], contour[:,0], color='green')

def bbox(seg):
    """Returns the bounding box of a segmentation mask

    Args:
        seg (np.array): segmentation mask array 2D

    Returns:
        ymin, ymax, xmin, xmax (int): bounding box coordinates
    """
    rows = np.any(seg, axis=0)
    cols = np.any(seg, axis=1)
    xmin, xmax = np.where(rows)[0][[0, -1]]
    ymin, ymax = np.where(cols)[0][[0, -1]]
    return ymin, ymax, xmin, xmax

def get_segment(mask:np.array, image:np.array, offset:int=0, make_zero:bool = False):
    #TODO:negative indexing must be prohibited
    ymin, ymax, xmin, xmax = bbox(mask)
    o = offset
    try:
        y0, y1 = ymin-o, ymax+1+o
        x0, x1 = xmin-o, xmax+1+o
        seg = np.copy(image[y0:y1, x0:x1])
        m   = np.copy(mask[ y0:y1, x0:x1])
        if seg.shape[0] == 0 or seg.shape[1] == 0: return None
        if make_zero:
            seg[m == 0] = 0
        return seg, m 
    except:
        return None

def make_training_set(stack: np.array, mask:np.array, offset:int, make_zero:bool = False, plot:bool = False):
    """script that makes a training set from a stack and a mask


    Args:
        stack (np.array): stack array 2D
        mask (np.array): array 2D of the mask
        offset (int): pixel padding around the mask
        plot (bool, optional): selects if to show each cell's output. Defaults to False.

    Returns:
        list: list of images
    """
    imin, imax = 1, np.max(mask)
    cells = []
    print(f'Number of estimated cells: {imax}')
    for i in range(imin, imax+1):
        m = mask == i
        # if np.sum(np.sum(m[m != 0]))>10:
        output = get_segment(m, stack, offset = offset, make_zero=make_zero)
        if output is not None:
            cell, m = output
            cells.append(cell)
            if plot:
                plt.title(i)
                plt.imshow(cell, cmap='gray')
                draw_contours(m)
                plt.show()
    return cells