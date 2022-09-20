from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os
from global_settings import correspondances_table

class DataSet(Dataset):

    def __init__(self, folder, mask=False) -> None:
        super().__init__()
        self.folder = folder
        self.mask = mask
        self._check_folder(folder)
        self.size = self.get_size()
        print(f'{self.size = }')
        self.classes = []
        self.shape = None
        self.__getitem__(0)

    def _check_folder(self, folder):
        if not os.path.exists(folder):
            raise OSError('Folder not found')
        if len(os.listdir(folder)) == 0:
            raise OSError('Folder is empty')

    def get_size(self):
        images = filter(lambda x: x.endswith('npy'), os.listdir(self.folder))
        N = len(list(images))
        return N

    def generate_mask(self, index, shape):
        df = pd.read_csv(f'{self.folder}/stack_{index}.csv')
        self.classes = [name for name in correspondances_table if 'Background' not in name]
        N_classes = len(self.classes)
        mask = np.zeros([N_classes+1, *shape])

        mask[-1] = 1 # Background generation
        for i, name in enumerate(self.classes):
            y = df[df['class'] == name]['axis-0']
            x = df[df['class'] == name]['axis-1']
            z = df[df['class'] == name]['axis-2']

            mask[i, y, x, z] = 1
            mask[-1] -= mask[i]

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
        # image = torch.tensor(image).float()
        # mask = torch.tensor(mask).float()
        return image, mask

class DatasetNetwork(DataSet):

    def __init__(self, folder) -> None:
        # super().__init__(folder, mask=True)
        self.folder = folder
        self._check_folder(folder)
        self.get_file_lists()
        self.size = self.get_size()
        self.get_shape()

    def get_file_lists(self):
        self.stack_list = sorted([file for file in os.listdir(self.folder) if 'stack' in file])
        self.mask_list  = sorted([file for file in os.listdir(self.folder) if 'mask'  in file])

    def get_size(self):
        images = filter(lambda x: x.endswith('npy') and 'stack' in x, os.listdir(self.folder))
        N = len(list(images))
        return N

    def normalize(self, arr):
        arr = arr.astype(float)
        if arr.max() != 0:
            arr /= arr.max()
        return arr

    def to_tensor(self, arrays:tuple):
        tensors = [torch.tensor(arr).float() for arr in arrays]
        return tensors

    def get_shape(self):
        stack, mask = self.__getitem__(0)
        self.number_of_classes = mask.shape[0]
        self.shape = stack.shape

    def __getitem__(self, index):
        stack_fname = self.stack_list[index]
        mask_fname = self.mask_list[index]

        stack = np.load(f'{self.folder}/{stack_fname}')
        mask  = np.load(f'{self.folder}/{mask_fname}')
        stack = np.expand_dims(stack, axis=0)
        stack = self.normalize(stack)
        stack, mask = self.to_tensor([stack, mask])
        return stack, mask

if __name__ == '__main__':
    pass
    # ds = DatasetNetwork('data/train')

    # for stack, mask in zip(ds.stack_list, ds.mask_list):
    #     if stack.split('stack')[1] != mask.split('mask')[1]:
    #         text = f'not a good list {stack} != {mask}'
    #         raise ProcessLookupError(text)
    #     print(stack, mask)
        