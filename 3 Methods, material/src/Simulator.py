from copy import copy
from CellSimulator import Cell
import numpy as np
from skimage import transform, filters
import matplotlib.pyplot as plt
import tqdm


class Simulator:

    def __init__(self,  canvas_shape,
                        list_of_cells,
                        cell_number,
                        dx, dy, dz,
                        sigma,
                        noise_signal_level,
                        noise_background_level):

        self.canvas_shape = canvas_shape
        self.list_of_cells = list_of_cells
        self.cell_number = cell_number
        self.dx = dx; self.dy = dy; self.dz = dz
        self.sigma = sigma
        self.noise_signal_level = noise_signal_level
        self.noise_background_level = noise_background_level
        self.image = np.zeros(canvas_shape)
        self.mask  = np.zeros(canvas_shape)
        

    def normalize(func):
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.image -= self.image.min()
            self.image /= self.image.max()
        return wrapper

    def run(self):
        self.populate()
        self.apply_filter()
        self.apply_noise()       
        return self.image

    def _expand(self, arr):
        h, w = self.canvas_shape
        py = (0, h - arr.shape[0])
        px = (0, w - arr.shape[1])
        arr = np.copy(arr)
        arr = np.pad(arr, (py, px), 'constant', constant_values=0)
        return arr

    def _shift(self, image, vector):
        tr = transform.AffineTransform(translation=vector)
        shifted = transform.warp(image, tr, preserve_range=True)
        shifted = shifted.astype(image.dtype)
        return shifted

    def position_cell(self, cell, pos):
        mask = self._expand(cell.mask_cytoplasm)
        mask = self._shift(mask, pos)
        return mask
    
    @normalize
    def apply_filter(self):
        self.image = filters.gaussian(self.image, sigma=self.sigma)

    @normalize
    def apply_noise(self):
        self.image *= (np.random.random(self.canvas_shape)+self.noise_signal_level)
        self.image += np.random.random(self.canvas_shape)*self.noise_background_level

    @normalize
    def populate(self):
        for i in tqdm.trange(self.cell_number):
            C = np.random.choice(self.list_of_cells)
            C.run()
            terminate = 0
            while True:
                y = np.random.randint(0, self.canvas_shape[0])
                x = np.random.randint(0, self.canvas_shape[1])
                mask = self.position_cell(C, (-y, -x))
                if np.any(self.mask[mask]):
                    terminate += 1
                else:
                    image = self._expand(C.image)
                    image = self._shift(image, (-y, -x))
                    self.mask += mask
                    self.image[mask] = image[mask]
                    break
                if terminate == 100:
                    raise RuntimeError('Could not position cell, try to decrease cell size or increase simulation size')

if __name__ == '__main__':
    dx = dy = dz = 0.5
    C1 = Cell(  
        size_cytoplasm          = 20,
        size_nucleus            = 10,
        dx = dx, dy = dy, dz = dz,
        gamma_nucleus           = 1.5,
        circ_nucleus            = 2,
        gamma_cytoplasm         = 0.5,
        circ_cytoplasm          = 2.0,
        gamma_cytoplasm_texture = .5,
        lowest_intensity        = 0.5)

    C2 = Cell(  
        size_cytoplasm          = 30,
        size_nucleus            = 15,
        dx = dx, dy = dy, dz = dz,
        gamma_nucleus           = 1.5,
        circ_nucleus            = 2.5,
        gamma_cytoplasm         = 1.0,
        circ_cytoplasm          = 2.0,
        gamma_cytoplasm_texture = .5,
        lowest_intensity        = 0.5)

    sim = Simulator(canvas_shape = (1000,1000),
                    list_of_cells = [C1, C2],
                    cell_number = 100,
                    dx = dx, dy = dy, dz = dz,
                    sigma = 2,
                    noise_signal_level = 0.5,
                    noise_background_level = 0.2)
    sim.run()
    plt.figure(figsize =(12,12))

    plt.imshow(sim.image)
    plt.colorbar()
    plt.show()
