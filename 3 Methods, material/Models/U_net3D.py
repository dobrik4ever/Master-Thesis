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
        self.BN = nn.BatchNorm3d(c_in)

    def forward(self, x):
        x = self.BN(x)
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
        self.BN = nn.BatchNorm3d(c_in)


    def forward(self, x):
        y = nn.functional.interpolate(x, scale_factor=MPS, mode='trilinear')
        y = self.BN(y)
        y = self.upconv_1(y);   
        y = self.upconv_2(y);   
        return y

class U_net3D(BaseModel):

    def __init__(self, img_size:tuple[int, int, int], learning_rate:float=0.01, number_of_classes:int=3, class_weights:tuple=None, batch_size:int=1):
        super().__init__(img_size, learning_rate)
        self.hparams.number_of_classes = number_of_classes
        self.hparams.batch_size = batch_size
        self.hparams.img_size = img_size
        self.hparams.class_weights = class_weights
        self.class_weights = torch.tensor(class_weights).float().cuda()
        self.save_hyperparameters()
        self.example_input_array = torch.unsqueeze(torch.rand(img_size), dim=0)
        self.setup_architecture()
    
    def setup_architecture(self):
        self.enc_1 = Encoder(1,   9, 3)
        self.enc_2 = Encoder(9,  18, 3) 
        self.enc_3 = Encoder(18, 36, 3)
        self.enc_4 = Encoder(36, 72, 3)
        self.enc_5 = Encoder(72, 144, 3)

        self.dec_5 = Decoder(144, 72, 3)
        self.dec_4 = Decoder(72, 36, 3)
        self.dec_3 = Decoder(36, 18, 3)
        self.dec_2 = Decoder(18,  9, 3)
        self.dec_1 = Decoder(9, self.hparams.number_of_classes, 3)

        self.output_function = nn.Softmax(dim=1)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def loss(self, output, target):
        loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)(output,target)
        return loss 

    def forward(self, x):
        ey1 = self.enc_1(x)
        ey2 = self.enc_2(ey1)
        ey3 = self.enc_3(ey2)
        # ey4 = self.enc_4(ey3)
 
        # dy4 = self.dec_4(ey4)
        dy3 = self.dec_3(ey3)
        dy2 = self.dec_2(dy3 + ey2)
        dy1 = self.dec_1(dy2 + ey1)
        y = self.output_function(dy1)
        return y