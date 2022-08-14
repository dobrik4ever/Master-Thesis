'''This script is used to set the global settings for the project.
    Run it to check if the configuration is correct.'''

import numpy as np

# Parameters
stack_size = [500, 500, 100]
chunk_size = [100, 100, 100]
dx = dy = 0.5
dz = 0.5

# Sanity check
def sanity_check():
    stackSize = np.array(stack_size)
    chunkSize = np.array(chunk_size)
    if np.any(stackSize % chunkSize != 0):
        text = f'Configuration is incorrect: stack could not be divided in chunks, given {stack_size = } {chunk_size = }'
        raise ValueError(text)

    print('Configuration is correct!')

if __name__ == '__main__':
    sanity_check()