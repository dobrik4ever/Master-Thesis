from Models import BaseModel
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from utils import Dataset
MPS = 2
class Encoder(torch.nn.Module):
    def __init__(self, c_in, c_out, ksize):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(c_in, c_out, ksize, padding=ksize // 2 + 1),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(c_out, c_out, ksize, padding=ksize // 2 + 1),
            nn.LeakyReLU(),
         )
        self.pool = nn.MaxPool3d(MPS)

    def forward(self, x):
        y = self.conv_1(x); 
        y = self.conv_2(y); 
        y = self.pool(y);
        return y

class Decoder(torch.nn.Module):

    def __init__(self, c_in, c_out, ksize):
        super().__init__()

        # self.unpool = nn.MaxUnpool3d(3)
        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose3d(c_in, c_out, ksize, padding=ksize // 2 + 1),
            nn.LeakyReLU()
        )
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose3d(c_out, c_out, ksize, padding=ksize // 2 + 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # y = self.unpool(x, ind);
        y = nn.functional.interpolate(x, scale_factor=MPS, mode='trilinear')
        y = self.upconv_1(y);   
        y = self.upconv_2(y);   
        return y

class U_net3D(BaseModel):

    def __init__(self, img_size, learning_rate=0.01, number_of_classes=3):
        super().__init__(img_size, learning_rate)
        self.hparams.number_of_classes = number_of_classes
        self.hparams.img_size = img_size
        self.enc_1 = Encoder(1,   9, 3)
        self.enc_2 = Encoder(9,  18, 3)
        self.enc_3 = Encoder(18, 36, 3)

        self.dec_3 = Decoder(36, 18, 3)
        self.dec_2 = Decoder(18,  9, 3)
        self.dec_1 = Decoder(9, number_of_classes, 3)

        self.output_function = nn.Softmax(dim=1)
        self.save_hyperparameters()
        self.example_input_array = torch.unsqueeze(torch.rand(img_size), dim=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_class_weights(self, x):
        weights = []
        all_ones = torch.sum(x == 1)
        for c in range(self.hparams.number_of_classes):
            ones_a = all_ones / torch.sum(x[:,c]==1) 
            if ones_a == torch.inf: ones_a = 1
            weights.append(ones_a)
        return torch.tensor(weights).float().cuda()

    def loss(self, output, target):
        # target = target[0]
        # loss_1 = torch.mean(torch.square(output[target == 0] - target[target == 0]))
        # if torch.sum(target == 1) == 0:
        #     loss_2 = 0
        # else:
        #     loss_2 = torch.mean(torch.square(output[target == 1] - target[target == 1]))
        # loss = loss_1 + loss_2
        # entries = target == 1
        # if torch.sum(entries) == 0:
            # loss = None
        # else:
            # loss = - torch.log(output[entries]).mean() # Softmax loss
        # zeros = torch.sum(target == 0)
        # ones = torch.sum(target == 1)
    
        o = output#[target==1]
        t = target#[target==1]
        if torch.sum(t == 1) == 0:
            loss = None
            return loss

        weights = self.compute_class_weights(t)
        loss = torch.nn.CrossEntropyLoss(weight=weights)(o,t)
        # loss_noweights = torch.nn.CrossEntropyLoss()(o,t)
        return loss 

    def forward(self, x):
        ey1 = self.enc_1(x)
        ey2 = self.enc_2(ey1)
        ey3 = self.enc_3(ey2)
 
        dy3 = self.dec_3(ey3)
        dy2 = self.dec_2(dy3 + ey2)
        dy1 = self.dec_1(dy2 + ey1)
        y = self.output_function(dy1)
        return y