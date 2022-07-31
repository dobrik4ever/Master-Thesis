from ast import excepthandler
from copy import copy
from src import Macrophage, T_Cell
import numpy as np
from skimage import transform, filters
import matplotlib.pyplot as plt
import tqdm
try:
    import napari
except ImportError:
    print('Could not import napari. Skipping visualization.')
    pass

class Simulator3D:
    cell_pos_tries_number = 1000

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

    def position_cell(self, cell, pos):
        mask = self._expand(cell.mask_cytoplasm)
        mask = self._shift(mask, pos)
        return mask

    def show(self):
        viewer = napari.Viewer()
        viewer.add_image(self.image, colormap='gray', rendering='average')
        viewer.dims.ndisplay = 3
        napari.run()
    
    @normalize
    def apply_filter(self):
        self.image = filters.gaussian(self.image, sigma=self.sigma)

    @normalize
    def apply_noise(self):
        self.image += np.random.random(self.canvas_shape)*self.noise_background_level
        self.image *= (np.random.random(self.canvas_shape)+self.noise_signal_level)

    @normalize
    def populate(self):
        for i in tqdm.trange(self.cell_number):
            C = np.random.choice(self.list_of_cells)
            C.run()
            terminate = 0
            while True:
                H, W, D = self.canvas_shape
                h, w, d = C.cell_shape

                if np.any(np.array(self.canvas_shape) - np.array(C.cell_shape) <= 0):
                    raise Exception('Cell is too big for canvas, increase canvas size.')

                y = np.random.randint(0, H - h)
                x = np.random.randint(0, W - w)
                z = np.random.randint(0, D - d)

                ys = H - h - y
                xs = W - w - x
                zs = D - d - z
                mask = np.pad(C.mask_cytoplasm, ((y,ys), (x,xs), (z,zs)), 'constant', constant_values=0)
                
                if np.any(self.mask[mask]):
                    terminate += 1
                else:
                    self.mask += mask
                    self.image[mask] = C.image[C.mask_cytoplasm]
                    break
                if terminate == self.cell_pos_tries_number:
                    raise RuntimeError('Could not position cell, try to decrease cell size or increase simulation size')

if __name__ == '__main__':
    dx = dy = dz = 0.5
    C1 = Macrophage(dx, dy, dz)
    C2 = T_Cell(dx, dy, dz)

    sim = Simulator3D(canvas_shape = (300,300, 100),
                    list_of_cells = [C1, C2],
                    cell_number = 10,
                    dx = dx, dy = dy, dz = dz,
                    sigma = 2,
                    noise_signal_level = 0.5,
                    noise_background_level = 0.2)
    sim.run()
    sim.show()
