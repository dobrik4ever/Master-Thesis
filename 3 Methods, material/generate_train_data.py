import os
import tqdm
import numpy as np
import pandas as pd
from utils import Chunkizer, Dataset

raw_folder   = 'data/raw'
train_folder = 'data/train'

chunk_shape = [100,100,100]

dataset = Dataset(raw_folder)
chunkizer = Chunkizer(
    input_shape=[1000,1000,100],
    chunk_shape=[100,100,100]   )

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



