import pytorch_lightning as pl
from Models import U_net3D
import torch
from utils import DatasetNetwork, SavingOutputCallback, calculate_class_weights

train_folder = 'data/train'
batch_size = 5
learning_rate = 1e-4

dataset = DatasetNetwork(train_folder)
class_weights = calculate_class_weights(train_folder)

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
model = U_net3D(
    img_size = dataset.shape,
    learning_rate=learning_rate,
    number_of_classes=dataset.number_of_classes,
    class_weights=class_weights
    )

model.hparams.batch_size = batch_size

trainer = pl.Trainer(
    gpus=-1,
    # overfit_batches=1,
    fast_dev_run = False,
    # accumulate_grad_batches=5,
    # resume_from_checkpoint='lightning_logs/version_31/checkpoints/epoch=592-step=2372.ckpt',
    log_every_n_steps=1,
    callbacks=[SavingOutputCallback()],
    )
trainer.fit(model, train_loader, valid_loader)