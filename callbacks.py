import os
import logging

import numpy as np
import pytorch_lightning as pl
import wandb

class LogWeightsCallback(pl.Callback):
    def __init__(self, checkpoints=None, layers=None):
        super().__init__()

        if checkpoints is None:
            self.checkpoints = []
        else:
            self.checkpoints = checkpoints
        
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        
    
    def on_train_batch(self, trainer, pl_module):
        if pl_module.global_step in self.checkpoints:
            path = os.path.join(wandb.run.dir, pl_module.global_step, 'weights.npz')
            logging.info(f'Logging weights after {pl_module.global_step} steps to {path}')
            self.save_weights(trainer, pl_module, path)
    
    def save_weights(self, trainer, pl_module, path):
        weight_dict = {}
        for layer in layers:
            weights = getattr(pl_module, layer, default=None)
            if weights is None:
                raise ValueError(f'pl_module does not have a layer called {layer}! Here is a summary of the module: {pl_module}')
            else:
                weights = weights.weight.data().cpu().numpy()
            weight_dict[layer] = weights
        np.savez_compressed(path, **weight_dict)
