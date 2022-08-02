from ast import excepthandler
from copy import copy
from src import Macrophage, T_Cell
import numpy as np
from skimage import transform, filters
import matplotlib.pyplot as plt
import tqdm
from numba import njit
import pandas as pd

try:
    import napari
except ImportError:
    print('Could not import napari. Skipping visualization.')
    pass

class Simulator3D:
    #TODO: write description
    cell_pos_tries_number = 1000

    def __init__(self,  canvas_shape,
                        list_of_cell_types,
                        cell_number,
                        sigma,
                        noise_signal_level,
                        noise_background_level):

        self.canvas_shape = canvas_shape
        self.list_of_cells_types = list_of_cell_types
        self.cell_number = cell_number
        self.sigma = sigma
        self.noise_signal_level = noise_signal_level
        self.noise_background_level = noise_background_level

        self.N_classes = len(list_of_cell_types)
        self.coordinates = {}
        for cell in self.list_of_cells_types:
            self.coordinates[cell] = []
        self.image = np.zeros(canvas_shape)
        self.mask  = np.zeros(canvas_shape)
        self.output_mask = None
        self.points      = None
        self.df_points   = None
        
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
        cmaps = ['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'bop blue', 'bop orange', 'bop purple']
        viewer = napari.Viewer()
        viewer.add_image(self.image, colormap='gray', blending='additive', rendering='average')

        # Visualize points of cells
        df = self.get_points_csv()
        for i, cell in enumerate(self.list_of_cells_types):
            CLASS = cell.__class__.__name__
            points = df[df['class'] == CLASS].drop('class', axis=1)
            viewer.add_points(points, name=CLASS, face_color=cmaps[i], size=3)

        viewer.dims.ndisplay = 3
        napari.run()

    def get_points_csv(self):
        df = {'class':[], 'axis-0': [], 'axis-1': [], 'axis-2': []}
        for key in self.coordinates:
            for coordinate in self.coordinates[key]:
                df['class'].append(key.__class__.__name__)
                df['axis-0'].append(coordinate[0])
                df['axis-1'].append(coordinate[1])
                df['axis-2'].append(coordinate[2])
        df = pd.DataFrame(df)
        self.df_points = df
        return self.df_points

    def get_point_mask(self):
        output_mask = np.zeros((self.N_classes, *self.image.shape))
        for key in self.coordinates:
            class_index = self.list_of_cells_types.index(key)
            for coordinate in self.coordinates[key]:
                y, x, z = coordinate
                output_mask[class_index][y, x, z] = 1
        self.output_mask = output_mask

    def save(self, path, name, tif=False):
        # TODO: save image and mask
        if not self.output_mask: self.get_point_mask()
        np.save(f'{path}/{name}.npy', self.image)

        if tif:
            from skimage import io
            io.imsave(f'{path}/{name}.tif', self.image)

        if not self.df_points: self.get_points_csv()
        self.df_points.to_csv(f'{path}/{name}.csv')
    
    @normalize
    def apply_filter(self):
        self.image = filters.gaussian(self.image, sigma=self.sigma)

    @normalize
    def apply_noise(self):
        self.image += np.random.random(self.canvas_shape)*self.noise_background_level
        self.image *= (np.random.random(self.canvas_shape)+self.noise_signal_level)

    @normalize
    def populate(self):
        pbar = tqdm.trange(self.cell_number)
        for i in pbar:
            cell = np.random.choice(self.list_of_cells_types)
            cell.run()
            terminate = 0
            while True:
                H, W, D = self.canvas_shape
                h, w, d = cell.cell_shape

                if np.any(np.array(self.canvas_shape) - np.array(cell.cell_shape) <= 0):
                    raise Exception('Cell is too big for canvas, increase canvas size.')

                y = np.random.randint(0, H - h)
                x = np.random.randint(0, W - w)
                z = np.random.randint(0, D - d)

                ys = H - h - y
                xs = W - w - x
                zs = D - d - z
                mask = np.pad(cell.mask_cytoplasm, ((y,ys), (x,xs), (z,zs)), 'constant', constant_values=0)
                
                if np.any(self.mask[mask]):
                    terminate += 1
                else:
                    self.mask += mask
                    self.image[mask] = cell.image[cell.mask_cytoplasm]
                    pos = np.array([y, x, z]) + cell.center
                    self.coordinates[cell].append(pos)
                    break
                if terminate == self.cell_pos_tries_number:
                    print('WARNING!: Could not position cell, try to decrease cell size or increase simulation size')
                    print(f'Total cell number is: {i}')
                    break
                    # raise RuntimeError('Could not position cell, try to decrease cell size or increase simulation size')
            if terminate == self.cell_pos_tries_number:
                break
        pbar.close()

if __name__ == '__main__':
    dx = dy = dz = 0.5
    C1 = Macrophage(dx, dy, dz)
    C2 = T_Cell(dx, dy, dz)

    sim = Simulator3D(canvas_shape = (100, 100, 100),
                    list_of_cell_types = [C1, C2],
                    cell_number = 10,
                    dx = dx, dy = dy, dz = dz,
                    sigma = 2,
                    noise_signal_level = 0.5,
                    noise_background_level = 0.2)
    sim.run()
    sim.show()
