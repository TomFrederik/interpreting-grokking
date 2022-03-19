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

interesting_keys = [
    5,18,24,25,26,36,41,49,59,65,70,71,73,74,76,
    77,81,82,84,98,101,105,110,119,146,153,165,175,
    177,182,211,217,220,232,241,249
]

for i in interesting_keys:
    lesioned_keys = keys.clone()
    lesioned_keys[:,[76,217,175]] = 0
    # lesioned_keys[:,[76,25]] = 0
    lesioned_probs = output_layer(model.transformer[0].norm2(value_layer(lesioned_keys) + normed_after_attn)).softmax(dim=-1)[torch.arange(len(data)),data[:,-1]]

    print(data[torch.argsort(lesioned_probs, dim=0)][:5,[0,2,4]])
    
    plt.figure()
    lesioned_probs[lesioned_probs>0.5] = 1
    plt.imshow(einops.rearrange(lesioned_probs.detach().cpu().numpy(), '(h w) -> h w', h=97), origin='lower')
    markers = [[0,89],[0,76],[0,66],[0,61],[0,43],[0,19],[0,20],[0,2],[0,7],[0,93],[0,25],[0,30],[0,53],[0,48],[0,71],[0,38],[0,15],[0,94]]
    markers = [[(m[0]+i)%97, (m[1]+i)%97] for i in range(97) for m in markers]
    plt.scatter(*zip(*markers), marker='x', s=3, c='r')
    plt.colorbar()
    plt.xticks(np.arange(0,97,10))
    plt.yticks(np.arange(0,97,10))
    plt.title(f"Lesioned key {i}")
    plt.show()
    plt.close()