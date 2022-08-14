import numpy as np
import torch
from utils import Chunkizer
import tqdm

class ModelFeeder:

    def __init__(self, model, chunk_size:tuple=[100,100,100]):
        self.model = model.cuda()
        self.chunk_size = chunk_size
        self.number_of_classes = self.model.hparams.number_of_classes
        
    def feed(self, stack):
        self.chunkizer = Chunkizer(stack.shape, chunk_shape=self.chunk_size)
        MAP = self.chunkizer.divide(stack)
        OUT = [np.zeros_like(MAP) for c in range(self.number_of_classes)]

        I,J,K = self.chunkizer.indices
        pbar = tqdm.tqdm(total=I*J*K)
        pbar.set_description('Feeding network with chunks')
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    chunk = MAP[i,j,k]
                    chunk = torch.tensor(chunk).float().cuda()
                    x = torch.unsqueeze(chunk, dim=0)
                    x = torch.unsqueeze(x, dim=0)
                    y = self.model.forward(x)
                    for c in range(self.number_of_classes):
                        OUT[c][i,j,k] = y.cpu().detach().numpy()[0, c]
                    pbar.update(1)
        pbar.close()
        stacks = []
        for c in range(self.number_of_classes):
            stacks.append(self.chunkizer.assemble(OUT[c]))
        self.output_stacks = stacks
        return self.output_stacks

    def save(self, fname:str):
        for c in range(self.number_of_classes):
            output = self.output_stacks[c]
            name = f'{fname}-class_{c}.npy'
            np.save(name, output)
            # print(f'Saved {name}')