import logging

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')

model_name = 'No Norm, Single Layer'
ckpt, ckpt_dir = load_model(model_name)

model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
model.eval()

key_layer = model.transformer[0].linear_net[0].weight
value_layer = model.transformer[0].linear_net[-1].weight

# interesting_keys = [309, 381, 449,  12,  77, 197, 300, 176, 463, 487, 414, 46, 104, 134]
# head 4 strong comp:
interesting_keys = [0,4,29,64,69,109,133,140,160,162,164,184,191,204,212,223,224,229,282,283,293,313,314,347,349,369,376,390,431,444,458,459]

# base_value = value_layer[:,interesting_keys[0]].detach().cpu().numpy()
# interesting_keys = list(range(512))
base_value = value_layer[:,444].detach().cpu().numpy()


all_values = value_layer[:,interesting_keys].detach().cpu().numpy()
dot_prod = np.dot(base_value, all_values)
dot_prod /= np.linalg.norm(base_value) * np.linalg.norm(all_values, axis=0)
sorted_idcs = np.argsort(dot_prod)[::-1][:]
key_dot = np.dot(key_layer[444,:].detach().cpu().numpy(), key_layer[interesting_keys,:].detach().cpu().numpy().T)
key_dot /= np.linalg.norm(key_layer[444,:].detach().cpu().numpy()) * np.linalg.norm(key_layer[interesting_keys,:].detach().cpu().numpy(), axis=1)
print(np.array(interesting_keys)[np.argsort(key_dot)[::-1]])
sorted_norms = np.linalg.norm(all_values, axis=0)[sorted_idcs]
sorted_cossim = dot_prod[sorted_idcs]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
axes[0].bar(np.arange(len(sorted_cossim)), sorted_cossim[:])
for i in range(len(sorted_cossim)):
    axes[0].text(*(i-0.25, sorted_cossim[i]+0.01), f"{sorted_norms[i]:.3f}", c='r')
axes[0].set_title('Values')


all_keys = key_layer[interesting_keys, :].detach().cpu().numpy()
base_key = all_keys[0]
dot_prod = np.dot(all_keys, base_key)
dot_prod /= np.linalg.norm(base_key) * np.linalg.norm(all_keys, axis=1)
sorted_idcs = np.argsort(dot_prod)[::-1]#[:10]
sorted_norms = np.linalg.norm(all_keys, axis=1)[sorted_idcs]
sorted_cossim = dot_prod[sorted_idcs]
axes[1].bar(np.arange(len(sorted_cossim)), sorted_cossim[:])
for i in range(len(sorted_cossim)):
    axes[1].text(*(i-0.25, sorted_cossim[i]+0.01), f"{sorted_norms[i]:.3f}", c='r')
axes[1].set_title('Keys')
plt.suptitle('Cossimilarities - Vector magnitude in red')
plt.grid()
plt.show()
# plt.savefig(f"{model_name}/mlp_cossims.png")

# the first six value vectors (including itself) have >99.9% cosinesimilarity with the first value vector