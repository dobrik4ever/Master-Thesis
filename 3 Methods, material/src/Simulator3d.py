from ast import excepthandler
from copy import copy
from src import Macrophage, T_Cell, Crypt3D
import numpy as np
from skimage import transform, filters, morphology
import matplotlib.pyplot as plt
import tqdm
import os
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
        """Simulator of 3d cytoplasm FAD fluorescence

        Args:
            canvas_shape (tuple:int): tuple, contains 3 entries, of stack size Y, X, Z
            list_of_cell_types (list:Cell3D): list contains Cell3D instances, will populate 
            canvas with selected cells in random manner
            cell_number (int): number of cells to be simulated. Will stop populate canvas if
            cell_pos_tries_number is exceeded during population.
            sigma (float): parameter for gaussian smoothing
            noise_signal_level (float): noise of photodetector
            noise_background_level (float): acquisition noise
        """

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
        viewer.add_image(self.image, name = 'image', colormap='gray', blending='additive', rendering='average')
        viewer.add_image(self.mask, name = 'mask', rendering='iso', opacity=0.5)

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

    def to_uint8(self, arr):
        arr /= arr.max()
        arr *= 255
        return arr.astype(np.uint8)

    def save(self, path, name, tif=False):
        fname = f'{path}/{name}.npy'
        if not self.output_mask: self.get_point_mask()
        if os.path.exists(fname): os.remove(fname)
        image = self.to_uint8(self.image)
        np.save(fname, image)

        if tif:
            from skimage import io
            fname = f'{path}/{name}.tif'
            if os.path.exists(fname): os.remove(fname)
            io.imsave(f'{path}/{name}.tif', image)

        if not self.df_points: self.get_points_csv()
        self.df_points.to_csv(f'{path}/{name}.csv')
    
    @normalize
    def apply_filter(self):
        self.image = filters.gaussian(self.image, sigma=self.sigma)

    @normalize
    def apply_noise(self):
        self.image += np.random.random(self.canvas_shape)*self.noise_background_level
        # self.image *= (np.random.random(self.canvas_shape)+self.noise_signal_level)

    @normalize
    def populate(self):
        pbar = tqdm.trange(self.cell_number, desc='Populating cells')
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

class Simulator3D_with_crypts(Simulator3D):
    crypt_size = 100
    crypt_number = 10
    crypt_spacing = 60
    cell_pos_tries_number = 1000
    def __init__(self,  canvas_shape,
                        list_of_cell_types,
                        cell_number,
                        sigma,
                        noise_signal_level,
                        noise_background_level,
                        dx, dy, dz):
        super().__init__(canvas_shape, list_of_cell_types, cell_number, sigma, noise_signal_level, noise_background_level)       
        self.dx = dx
        self.dy = dy
        self.dz = dz 

    def run(self):
        self.populate_crypts()
        self.populate()
        # self.apply_filter()
        # self.apply_noise()       
        return self.image

    def populate_crypts(self):
        pbar = tqdm.trange(self.crypt_number, desc='Populating crypts')
        crypt = Crypt3D(self.crypt_size, self.canvas_shape[2], self.crypt_spacing, self.dx, self.dy, self.dz)
        for i in pbar:
            crypt.run()
            terminate = 0
            H, W, D = self.canvas_shape
            h, w, d = crypt.mask.shape

            if np.any(np.array(self.canvas_shape) - np.array(crypt.crypt_shape) < 0):
                raise Exception('crypt is too big for canvas, increase canvas size.')

            self.mask2d = self.mask[:,:,0]
            mask2d = crypt.mask[:,:,0]
            while True:
                y = np.random.randint(0, H - h)
                x = np.random.randint(0, W - w)

                ys = H - h - y
                xs = W - w - x
                mask = np.pad(mask2d, ((y,ys), (x,xs)), 'constant', constant_values=0).astype(bool)

                if np.any(self.mask2d[mask]):
                    terminate += 1
                else:
                    self.mask2d += mask
                    break

                if terminate == self.cell_pos_tries_number:
                    print('WARNING!: Could not position crypt, try to decrease crypt size or increase simulation size')
                    print(f'Total crypt number is: {i}')
                    break
                    # raise RuntimeError('Could not position cell, try to decrease cell size or increase simulation size')
            if terminate == self.cell_pos_tries_number:
                break

        for i in range(1,self.mask.shape[2]):
            self.mask[:,:,i] = self.mask2d

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
