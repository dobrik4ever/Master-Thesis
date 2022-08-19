from utils import ModelFeeder
from Models import U_net3D
import numpy as np
import torch

stack = np.load('data/raw/stack_1.npy')

model = U_net3D(stack.shape)
succesfull = []
# feeder = ModelFeeder(model)
for i in range(0,200):
    tensor = torch.rand([1,i,i,i])
    print(f'Trying {i}:', end=' ')
    try:
        output = model.forward(tensor)
        print('successfull')
        succesfull.append(i)
    except Exception as e:
        print('failed')
    # break

print(f'List of successfull sizes: {succesfull}')
# output = feeder.feed(stack)
# feeder.save('model_output')