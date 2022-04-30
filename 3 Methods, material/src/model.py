from torch import nn
import torch
class U_net_01(torch.nn.Module):
    """A first attempt to reduce dimensionality of the cell
    """

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(1, 3, 7), nn.LeakyReLU(), nn.MaxPool2d(7, 1),
            nn.Conv2d(3, 3, 7), nn.LeakyReLU(), nn.MaxPool2d(7, 1),
            ])

    def forward(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    model = U_net_01()
    A = np.zeros([1,1,100,100]).astype(np.float32)
    A[0,0,20:40,20:40] = 1
    tt = torch.tensor(A)
    out = model.forward(tt).detach().numpy()[0,0]
    print(out.shape)
    plt.imshow(out)
    plt.show()