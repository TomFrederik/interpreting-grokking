import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Subset

from datasets import get_dataset
from hooking_utils import hook_model, get_model_layers
from model import GrokkingTransformer


# use the same settings as during training here, otherwise your results will be confused
seed = 42
train_ratio = 0.5
pl.seed_everything(seed)

# get data set
data = get_dataset(descr='minus', num_elements=97, data_dir='./data')
idcs = np.random.permutation(np.arange(len(data)))
train_idcs = idcs[:int(train_ratio * len(idcs))]
val_idcs = idcs[int(train_ratio * len(idcs)):]
train_data = Subset(data, train_idcs)
val_data = Subset(data, val_idcs)

steps = [50 + 50 * i for i in range(20)]
model_paths = [os.path.join("wandb/run-20211205_210604-1lnuqp96/files", f"{step}.ckpt") for step in steps]

for model_path in model_paths:
    model = GrokkingTransformer.load_from_checkpoint(model_path)
    model.eval()
    print(model.model)
    raise ValueError

# compute highest loss examples
x_train = torch.stack([torch.tensor(sample) for sample in train_data])
x_val = torch.stack([torch.tensor(sample) for sample in val_data])
# plt.figure()
# plt.scatter(x_train[:,0], x_train[:,2], s=7, alpha=.3, label='train')
# plt.scatter(x_val[:,0], x_val[:,2], s=7, alpha=.3, label='val')

# plt.figure()
# for i, step in enumerate(steps):
#     model = GrokkingTransformer.load_from_checkpoint(model_paths[i])
#     model.eval()
#     y_hat = model(x_val[:,:-1])
#     losses = torch.nn.CrossEntropyLoss(reduction='none')(y_hat[:,-1], x_val[:,-1])
#     correct = torch.argmax(y_hat[:,-1], dim=1) == x_val[:,-1]
#     x_not_correct = x_val[~correct]
#     x_correct = x_val[correct]
#     # plt.scatter(x_correct[:,0], x_correct[:,2], s=7, alpha=0.5, marker='+', label='correct', c='r')
#     ratio_mod = (x_not_correct[:, 0] < x_not_correct[:, 2]).sum() / x_not_correct.shape[0]
#     print(f"Step {step}: {ratio_mod.item():.4f}")
# plt.legend()
# plt.show()
# plt.show()    


# model = GrokkingTransformer.load_from_checkpoint(model_paths[0])
# model.eval()
# hook = hook_model(model, lambda: x_val[0])
# model(torch.stack([x_val[0]]))
# print(model.model)
# print(get_model_layers(model))
# print(hook('model_attn_layers_layers_2_1')[-1][-1])
# print(hook.features)
# print(hook.modules)