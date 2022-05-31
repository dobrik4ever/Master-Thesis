import os
from DataSimulation.CellSimulator import Simulator, Ellipse_cell
from skimage import io
from scipy.ndimage import filters
import numpy as np

import torch
import torchvision as tv
from torch.utils.data import Dataset
import tqdm

class DataGenerator(Dataset):
    number_of_classes = 1
    img_size = 300
    output_shape = 30 # output would be (B, 3 + NC, 60, 60)
    def __init__(self, folder, size):
        super().__init__()
        self.size = size
        self.folder = folder
        if not os.path.isdir(folder):
            os.mkdir(folder)
        else:
            for file in os.listdir(folder):
                os.remove(f'{folder}/{file}')

        self._transform = tv.transforms.Compose([
          tv.transforms.ToTensor()
        ])

    def generate(self):
        for i in tqdm.trange(self.size):
            BG = Simulator()
            BG.add_cells(Ellipse_cell, 100)
            BG.apply_filter(filters.gaussian_filter, args=(20,))
            BG.normalize()

            sim = Simulator()
            Ellipse_cell.cytoplasm_r = (7,15)
            Ellipse_cell.nucleus_r = (2,5)

            sim.add_cells(Ellipse_cell, 16)
            
            # Ellipse_cell.cytoplasm_r = (6,8)
            # Ellipse_cell.nucleus_r = (1,4)

            # sim.add_cells(Ellipse_cell, 16)
            sim.apply_filter(filters.gaussian_filter, args=(2,))
            sim.add_noise(.6)
            sim.normalize()

            sim.canvas = sim.canvas + BG.canvas
            sim.normalize()

            coordinates = coordinates_to_tensor(sim.X, sim.Y, self.output_shape, self.img_size)

            np.save(f'{self.folder}/img_{i}.npy', sim.canvas)
            np.save(f'{self.folder}/pos_{i}.npy', coordinates)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        canvas = np.load(f'{self.folder}/img_{index}.npy')
        coordinates = np.load(f'{self.folder}/pos_{index}.npy')
        return self._transform(canvas).float(), torch.tensor(coordinates).float()

def coordinates_to_tensor(Ix:np.array, Iy:np.array, S:int, I:int) -> np.array:
    """Utility function to convert coordinates to tensor

    Args:
        Ix (np.array 1d): array of x coordinates
        Iy (np.array 1d): array of y coordinates
        S (int): tensor XY size
        I (int): image XY size

    Returns:
        np.array: 3D tensor of shape (3, S, S), 0 - X, 1 - Y, 2 - objectness
    """

    Ci = Ix * S // I
    Cj = Iy * S // I

    Sx = Ix * S / I - Ci
    Sy = Iy * S / I - Cj

    array = np.zeros([3, S, S])
    array[0,Cj, Ci] = Sx
    array[1,Cj, Ci] = Sy
    array[2,Cj, Ci] = 1

    return array

def tensor_to_coordinates(tensor:np.array, S:int, I:int):
    """Utility function to convert tensor to coordinates

    Args:
        tensor (np.array): 3D tensor of shape (3, S, S), 0 - X, 1 - Y, 2 - objectness
        S (int): tensor XY size
        I (int): image XY size

    Returns:
        (x, y) tuple: 2 arrays of x and y coordinates
    """
        
    x, y = tensor[0] * I / S, tensor[1] * I / S

    xx, yy = np.meshgrid(np.arange(0, S), np.arange(0, S))
    xx[tensor[2] == 0] = 0
    yy[tensor[2] == 0] = 0
    
    Ix = xx * I // S + x
    Iy = yy * I // S + y

    return Ix, Iy
    