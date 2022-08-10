from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os

class Dataset(Dataset):

    def __init__(self, folder, mask=False) -> None:
        super().__init__()
        self.folder = folder
        self.mask = mask
        if not os.path.exists(folder):
            raise OSError('Folder not found')
        if len(os.listdir(folder)) == 0:
            raise OSError('Folder is empty')
        self.size = self.get_size()
        print(f'{self.size = }')
        self.classes = []
        self.shape = None
        # self.__getitem__(0)

    def get_size(self):
        images = filter(lambda x: x.endswith('npy'), os.listdir(self.folder))
        N = len(list(images))
        return N

    def generate_mask(self, index, shape):
        df = pd.read_csv(f'{self.folder}/stack_{index}.csv')
        self.classes = df['class'].unique()
        N_classes = len(self.classes)
        mask = np.zeros([N_classes, *shape])

        for i, name in enumerate(self.classes):
            y = df[df['class'] == name]['axis-0']
            x = df[df['class'] == name]['axis-1']
            z = df[df['class'] == name]['axis-2']

            mask[i, y, x, z] = 1

        return mask

    def __str__(self):
        text = ''
        text += 'DataLoader\nAxes: 0-Y, 1-X, 2-Z\n'
        classes = " ".join([f"{i}-{name}, " for i, name in enumerate(self.classes)])
        text += f'Classes: {classes}\n'
        text += f'Images number: {self.size}'
        return text

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = np.load(f'{self.folder}/stack_{index}.npy')
        self.shape = image.shape
        if self.mask:
            mask = np.load(f'{self.folder}/mask_{index}.npy')
        else:
            mask = self.generate_mask(index, image.shape)
        image = torch.tensor(image).float()
        mask = torch.tensor(mask).float()
        return image, mask
        