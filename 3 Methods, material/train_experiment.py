import pytorch_lightning as pl
from Models import U_net3D
import torch
from utils import DatasetNetwork, SavingOutputCallback, calculate_class_weights

train_folder = 'data/train'
batch_size = 5
# learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
learning_rate = 1e-3
class_weights = calculate_class_weights(train_folder)
SavingOutputCallback.save_every = 99

class U_net_1(U_net3D):
    def __init__(self, img_size: tuple[int, int, int], learning_rate: float = 0.01, number_of_classes: int = 3, class_weights: tuple = None, batch_size: int = 1):
        super().__init__(img_size, learning_rate, number_of_classes, class_weights, batch_size)

    def forward(self, x):
        ey1 = self.enc_1(x)
        ey2 = self.enc_2(ey1)
        ey3 = self.enc_3(ey2)
 
        dy3 = self.dec_3(ey3)
        dy2 = self.dec_2(dy3 + ey2)
        dy1 = self.dec_1(dy2 + ey1)
        y = self.output_function(dy1)
        return y

class U_net_2(U_net3D):
    def __init__(self, img_size: tuple[int, int, int], learning_rate: float = 0.01, number_of_classes: int = 3, class_weights: tuple = None, batch_size: int = 1):
        super().__init__(img_size, learning_rate, number_of_classes, class_weights, batch_size)

    def forward(self, x):
        ey1 = self.enc_1(x)
        ey2 = self.enc_2(ey1)
        ey3 = self.enc_3(ey2)
        ey4 = self.enc_4(ey3)
 
        dy4 = self.dec_4(ey4)
        dy3 = self.dec_3(dy4 + ey3)
        dy2 = self.dec_2(dy3 + ey2)
        dy1 = self.dec_1(dy2 + ey1)
        y = self.output_function(dy1)
        return y

class U_net_3(U_net3D):
    def __init__(self, img_size: tuple[int, int, int], learning_rate: float = 0.01, number_of_classes: int = 3, class_weights: tuple = None, batch_size: int = 1):
        super().__init__(img_size, learning_rate, number_of_classes, class_weights, batch_size)

    def forward(self, x):
        ey1 = self.enc_1(x)
        ey2 = self.enc_2(ey1)
        ey3 = self.enc_3(ey2)
        ey4 = self.enc_4(ey3)
        ey5 = self.enc_5(ey4)
 
        dy5 = self.dec_5(ey5)
        dy4 = self.dec_4(dy5 + ey4)
        dy3 = self.dec_3(dy4 + ey3)
        dy2 = self.dec_2(dy3 + ey2)
        dy1 = self.dec_1(dy2 + ey1)
        y = self.output_function(dy1)
        return y

models = [U_net_1, U_net_2, U_net_3]
# Testing
# for Model in models:
#     model = Model(
#     img_size = [100,100,100],
#     learning_rate=learning_rate,
#     number_of_classes=3,
#     class_weights=class_weights,
#     batch_size = batch_size,
#     )
#     model.summarize()
    
for net_i, net in enumerate(models):
    pl.utilities.seed.seed_everything(0)


    dataset = DatasetNetwork(train_folder)

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = net(
        img_size = dataset.shape,
        learning_rate=learning_rate,
        number_of_classes=dataset.number_of_classes,
        class_weights=class_weights,
        batch_size = batch_size,
        )

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=1000,
        # overfit_batches=1,
        fast_dev_run = False,
        # accumulate_grad_batches=5,
        # resume_from_checkpoint='lightning_logs/version_31/checkpoints/epoch=592-step=2372.ckpt',
        log_every_n_steps=1,
        callbacks=[SavingOutputCallback()],
        )

    trainer.fit(model, train_loader, valid_loader)
    trainer.save_checkpoint(f'checkpoints/{net_i}_{type(model).__name__}.ckpt')