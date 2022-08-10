from Models import BaseModel
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from utils import Dataset

class Encoder(torch.nn.Module):
    def __init__(self, c_in, c_out, ksize):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(c_in, c_out, ksize, padding=ksize // 2+1),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(c_out, c_out, ksize, padding=ksize // 2+1),
            nn.LeakyReLU(),
         )
        self.pool = nn.MaxPool3d(2, return_indices=True)

    def forward(self, x):
        y = self.conv_1(x); 
        y = self.conv_2(y); 
        y, z = self.pool(y);
        return y, z

class Decoder(torch.nn.Module):

    def __init__(self, c_in, c_out, ksize):
        super().__init__()

        self.unpool = nn.MaxUnpool3d(2)
        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose3d(c_in, c_out, ksize, padding=ksize // 2+1),
            nn.LeakyReLU()
        )
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose3d(c_out, c_out, ksize, padding=ksize // 2+1),
            nn.LeakyReLU()
        )

    def forward(self, x, ind):
        y = self.unpool(x, ind);
        y = self.upconv_1(y);   
        y = self.upconv_2(y);   
        return y

class U_net3D(BaseModel):

    def __init__(self, img_size, learning_rate, batch_size):
        super().__init__(img_size, learning_rate)
        self.batch_size = batch_size
        self.enc_1 = Encoder(1, 9, 3)
        self.enc_2 = Encoder(9, 18, 3)
        self.enc_3 = Encoder(18, 36, 3)

        self.dec_3 = Decoder(36, 18, 3)
        self.dec_2 = Decoder(18, 9, 3)
        self.dec_1 = Decoder(9, 1, 3)

        self.output_function = nn.LeakyReLU()
        self.save_hyperparameters()
        self.example_input_array = torch.rand([1,1, *img_size])

    def loss(self, output, target):
        loss_1 = torch.mean(torch.square(output[target == 0] - target[target == 0]))
        loss_2 = torch.mean(torch.square(output[target == 1] - target[target == 1]))
        loss = loss_1 + loss_2
        return loss 

    def forward(self, x):
        ey1, z1 = self.enc_1(x)
        ey2, z2 = self.enc_2(ey1)
        ey3, z3 = self.enc_3(ey2)
 
        dy3 = self.dec_3(ey3, z3)
        dy2 = self.dec_2(dy3 + ey2, z2)
        dy1 = self.dec_1(dy2 + ey1, z1)
        y = self.output_function(dy1)
        return y

    def train_dataloader(self):
        dl = Dataset('data/raw')
        loader = torch.utils.data.DataLoader(dl, batch_size=self.batch_size, shuffle=False)
        return loader