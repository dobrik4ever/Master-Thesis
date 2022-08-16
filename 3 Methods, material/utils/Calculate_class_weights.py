import numpy as np
import os
import tqdm

def calculate_class_weights(folder = 'data/train'):
    folder = 'data/train'
    file_list = sorted([file for file in os.listdir(folder) if 'mask' in file])

    mask = np.load(f'{folder}/{file_list[0]}')
    number_of_classes = mask.shape[0]

    all_ones = 0
    counts = {Class: 0 for Class in range(number_of_classes)}
    pbar = tqdm.tqdm(total=len(file_list))
    for file in file_list:
        mask = np.load(f'{folder}/{file}')
        for c in range(number_of_classes):
            ones = np.sum(mask == 1)
            class_count = np.sum(mask[c]==1)

            all_ones += ones
            counts[c] += class_count
        pbar.update(1)
    pbar.close()

    weights = [all_ones / counts[c] for c in counts]
    return weights

if __name__ == '__main__':
    weights = calculate_class_weights()
    print(f'{weights = }')


        