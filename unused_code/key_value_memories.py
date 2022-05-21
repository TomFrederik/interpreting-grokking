import logging

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')

model_name = 'Small MLP'
ckpt, ckpt_dir = load_model(model_name)

model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
model.eval()

key_layer = model.transformer[0].linear_net[0]
value_layer = model.transformer[0].linear_net[-1]
output_layer = model.output

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().to(device)
x = data[:,:-1]
x = model.pos_encoding(model.embedding(x))
context_len = x.shape[1]
attn_out, _ = model.transformer[0].self_attn(x, model.self_attn_mask[:context_len,:context_len], return_attention=True)

normed_after_attn = model.transformer[0].norm1(attn_out + x)[:,-1] # only look at embedding of equal sign

keys = torch.nn.functional.relu(key_layer(normed_after_attn))
print(f"{keys.shape = }") # 9409, 512
sorted_keys = keys.argsort(dim=0, descending=True)

# look at top 10 examples for key 0
top10_0 = sorted_keys[:20,0]
print(data[top10_0][:,[0,2,4]])
top10_1 = sorted_keys[:20,1]
print(data[top10_1][:,[0,2,4]])
print(data[keys[:,0] == 0])

fig, axes = plt.subplots(2,2, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10,10))
for i in range(4):
    col = i // 2
    row = i % 2
    axes[row,col].imshow(einops.rearrange(keys[:,i].detach().cpu().numpy(), '(h w) -> h w', h=97), origin='lower')
    axes[row,col].set_title(f"Key {i}")
plt.show()