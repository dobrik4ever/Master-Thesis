import pytorch_lightning as pl
import numpy as np
import os
from global_settings import correspondances_table


class SavingOutputCallback(pl.Callback):
    stack_folder = 'data/test'
    output_folder = 'data/output'
    stack_id = 1
    save_every = 1
    classes = [name for name in correspondances_table if name != 'Background']

    def __init__(self) -> None:
        super().__init__()
        self._empty_output_folder()

    def _empty_output_folder(self):
        for file in os.listdir(self.output_folder):
            os.remove(f'{self.output_folder}/{file}')

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        if epoch % self.save_every == 0 and epoch != 0:
            y_pred = pl_module.forward_from_file(f'{self.stack_folder}/stack_{self.stack_id}.npy')
            np.save(f'{self.output_folder}/{pl_module.__class__.__name__}_epoch_{epoch}.npy', y_pred)


