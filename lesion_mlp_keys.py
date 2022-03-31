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

model_name = 'Single Layer ReLU'
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
    11, 12, 14, 23, 25, 29, 44, 80, 81, 90, 113, 
    121, 140, 144, 149, 151, 155, 157, 161, 162, 
    166, 174, 189, 190, 199, 202, 204, 219, 222, 
    231, 233, 247, 249, 261, 263, 267, 268, 277, 
    283, 287, 294, 303, 305, 307, 309, 320, 330, 
    335, 342, 354, 355, 365, 374, 381, 403, 412, 
    446, 453, 459, 466, 476, 482, 491
]
interesting_keys = [8,14]

for i in interesting_keys:
    lesioned_keys = keys.clone()
    print(f"{lesioned_keys.shape = }")
    print(lesioned_keys[:,8])
    break
    lesioned_keys[:,i] = 0
    # lesioned_keys[:,[219, 268, 446]] = 0
    lesioned_probs = output_layer(model.transformer[0].norm2(value_layer(lesioned_keys) + normed_after_attn)).softmax(dim=-1)[torch.arange(len(data)),data[:,-1]]

    plt.figure()
    # lesioned_probs[lesioned_probs>0.5] = 1
    plt.imshow(einops.rearrange(lesioned_probs.detach().cpu().numpy(), '(h w) -> h w', h=97), origin='lower', norm=colors.LogNorm())
    # markers = [[0,89],[0,76],[0,66],[0,61],[0,43],[0,19],[0,20],[0,2],[0,7],[0,93],[0,25],[0,30],[0,53],[0,48],[0,71],[0,38],[0,15],[0,94]]
    # markers = [[(m[0]+i)%97, (m[1]+i)%97] for i in range(97) for m in markers]
    # plt.scatter(*zip(*markers), marker='x', s=3, c='r')
    plt.colorbar()
    plt.xticks(np.arange(0,97,10))
    plt.yticks(np.arange(0,97,10))
    # plt.title(f"Lesioned key {i}")
    # plt.title(f"Lesioned key [219, 268, 446]")
    plt.show()
    
    # plt.savefig(f"{model_name}/lesion_mlp_keys_{i}.jpg")
    # plt.savefig(f"{model_name}/lesion_mlp_keys_219_268_446.jpg")
    # break
    # plt.show()
    plt.close()
