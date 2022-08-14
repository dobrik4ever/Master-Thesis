import pytorch_lightning as pl
from Models import U_net3D
import torch
from utils import DatasetNetwork, SavingOutputCallback

# torch.utils
dataset = DatasetNetwork('data/train')

train_size = int(0.5 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
model = U_net3D(dataset.shape, learning_rate=1e-3, number_of_classes=dataset.number_of_classes)
trainer = pl.Trainer(
    gpus=-1,
    fast_dev_run = False,
    # overfit_batches=1,
    log_every_n_steps=1,
    callbacks=[SavingOutputCallback()],
    accumulate_grad_batches=6,
    # resume_from_checkpoint='lightning_logs/version_26/checkpoints/epoch=58-step=59.ckpt',
    )
trainer.fit(model, train_loader, valid_loader)