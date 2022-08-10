from random import shuffle
import pytorch_lightning as pl
from Models import U_net3D
import torch
from utils import Dataset


dl = Dataset('data/raw')
# train_loader = torch.utils.data.DataLoader(dl, batch_size=2, shuffle=False)
# model = U_net3D(dl.shape, learning_rate=0.01)
model = U_net3D(dl.shape, learning_rate=0.01, batch_size=1)
model.batch_size = 1
model.data_loader = dl
trainer = pl.Trainer(gpus=-1, fast_dev_run = False, auto_scale_batch_size=True)#, progress_bar_refresh_rate=20, callbacks=[PlottingCallback(train_loader, valid_loader)])
trainer.tune(model)
# print(model)