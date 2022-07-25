from CellSimulator import Cell
import numpy as np
from skimage import transform, filters
import matplotlib.pyplot as plt



class Simulator:

    def __init__(self, shape, cell_number) -> None:
        self.image = np.zeros(shape)
        self.mask  = np.zeros(shape)
        self.shape = shape
        self.cell_number = cell_number
        self.sigma = 2
        self.noise_signal_level = 0.5
        self.noise_background_level = 0.2

        self.C1 = Cell(  
            cytoplasm_size          = 20,
            nucleus_size            = 10,
            gamma_nucleus           = 1.5,
            circ_nucleus            = 2,
            gamma_cytoplasm         = 0.5,
            circ_cytoplasm          = 2.0,
            gamma_cytoplasm_texture = .5,
            lowest_intensity        = 0.5)

        self.C2 = Cell(  
            cytoplasm_size          = 10,
            nucleus_size            = 5,
            gamma_nucleus           = 1.5,
            circ_nucleus            = 2.5,
            gamma_cytoplasm         = 1.0,
            circ_cytoplasm          = 2.0,
            gamma_cytoplasm_texture = .5,
            lowest_intensity        = 0.5)

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
        h, w = self.shape
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
        self.image *= (np.random.random(self.shape)+self.noise_signal_level)
        self.image += np.random.random(self.shape)*self.noise_background_level

    @normalize
    def populate(self):
        for i in range(self.cell_number):
            C = np.random.choice([self.C1, self.C2])
            C.run()
            terminate = 0
            while True:
                y = np.random.randint(0, self.shape[0])
                x = np.random.randint(0, self.shape[1])
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
    sim = Simulator(shape = (100,100), cell_number = 5)
    sim.run()
    plt.figure(figsize =(12,12))

    plt.imshow(sim.image)
    plt.colorbar()
    plt.show()
