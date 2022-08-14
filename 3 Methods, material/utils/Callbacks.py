import torch
import pytorch_lightning as pl
import numpy as np
from utils import ModelFeeder
import os
from global_settings import chunk_size, stack_size

class SavingOutputCallback(pl.Callback):
    stack_folder = 'data/raw'
    output_folder = 'data/output'
    stack_id = 0
    save_every = 10
    def __init__(self) -> None:
        super().__init__()
        self._empty_output_folder()

    def _empty_output_folder(self):
        for file in os.listdir(self.output_folder):
            os.remove(f'{self.output_folder}/{file}')

    def normalize_array(self, arr):
        arr = arr.astype(float)
        # arr = np.expand_dims(arr, axis=0)
        if arr.max() != 0:
            arr /= arr.max()
        return arr

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        if epoch % self.save_every == 0:

            stack = np.load(f'{self.stack_folder}/stack_{self.stack_id}.npy')
            stack = self.normalize_array(stack)
            feeder = ModelFeeder(pl_module, chunk_size=chunk_size)
            feeder.feed(stack)
            feeder.save(f'data/output/ModelOutput_epoch_{epoch}')


