from turtle import forward
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import abc

class BaseModel(pl.LightningModule):
    DEBUG_SHAPE = True
    def __init__(self, img_size, learning_rate):
        super().__init__()
        self.img_size = img_size
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.forward(x)    
        loss = self.loss(y_pred, y)
        if loss != None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)    
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss)
        return loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))