import os
import tqdm
import numpy as np
import pandas as pd
from utils import Chunkizer, DataSet
from global_settings import stack_size, chunk_size

raw_folder   = 'data/raw'
train_folder = 'data/train'

dataset = DataSet(raw_folder)
print(dataset)
chunkizer = Chunkizer(
    input_shape=stack_size,
    chunk_shape=chunk_size
)

for file in tqdm.trange(dataset.size):
    stack, mask = next(iter(dataset))
    stack_map = chunkizer.divide(stack)
    mask_map  = chunkizer.divide(mask)
    I, J, K = chunkizer.indices
    for i in range(I):
        for j in range(J):
            for k in range(K):
                fname_stack = f'{train_folder}/stack_{file}_{i}-{j}-{k}.npy'
                fname_mask  = f'{train_folder}/mask_{file}_{i}-{j}-{k}.npy'
                np.save(fname_stack, stack_map[i,j,k])
                np.save(fname_mask,  mask_map[:,i,j,k])



