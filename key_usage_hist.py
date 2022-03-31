from collections import Counter
import itertools
import logging
import os

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors
from unseal.hooks import Hook, HookedModel
from unseal.hooks.common_hooks import save_output
from tqdm import tqdm 

from datasets import get_dataset
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')
    
model_name = 'Single Layer ReLU'
ckpt, ckpt_dir = load_model(model_name)

model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
model.eval()

key_layer = model.transformer[0].linear_net[0].weight
value_layer = model.transformer[0].linear_net[-1].weight

dataset = torch.from_numpy(get_dataset(descr='minus', num_elements=97, data_dir='./data').data).to(device)[:,:-1]

hooked_model = HookedModel(model)
hook = Hook("transformer->0->linear_net->2", save_output(), key="save_key_magnitude")
hooked_model(dataset, hooks=[hook])

key_magnitude = hooked_model.save_ctx['save_key_magnitude']['output'][:,-1,:]

os.makedirs(f'{model_name}/key_mag_hist/', exist_ok=True)
for i in tqdm(range(key_magnitude.shape[-1])):
    plt.figure()
    plt.hist(key_magnitude[:,i].numpy(), bins=100)
    plt.title(f"Key {i} Magnitude Histogram")
    plt.ylim(0,200)
    plt.savefig(f"{model_name}/key_mag_hist/key_{i}.png")
    plt.close()