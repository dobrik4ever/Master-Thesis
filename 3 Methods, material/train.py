import pytorch_lightning as pl
from Models import U_net3D
import torch
from utils import DatasetNetwork, SavingOutputCallback, calculate_class_weights

train_folder = 'data/train'
batch_size = 1
learning_rate = 1e-4

pl.utilities.seed.seed_everything(0)

dataset = DatasetNetwork(train_folder)
class_weights = calculate_class_weights(train_folder)

train_size = int(3/4 * len(dataset))
valid_size = len(dataset) - train_size
torch.manual_seed(0)
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

model = U_net3D(
    img_size = dataset.shape,
    learning_rate=learning_rate,
    number_of_classes=dataset.number_of_classes,
    class_weights=class_weights
    )

model.hparams.batch_size = batch_size

trainer = pl.Trainer(
    gpus=-1,
    max_epochs=50,
    fast_dev_run = False,
    log_every_n_steps=1,
    callbacks=[SavingOutputCallback()],
    )
trainer.fit(model, train_loader, valid_loader)