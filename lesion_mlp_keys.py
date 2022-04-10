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

key_layer = model.transformer[0].linear_net[0]
value_layer = model.transformer[0].linear_net[-1]
output_layer = model.output

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97, no_op_token=True, force_data=True).data).long().to(device)
x = data[:,:-1]
x = model.pos_encoding(model.embedding(x))
context_len = x.shape[1]
attn_out, _ = model.transformer[0].self_attn(x, model.self_attn_mask[:context_len,:context_len], return_attention=True)

normed_after_attn = model.transformer[0].norm1(attn_out + x)[:,-1] # only look at embedding of equal sign

keys = torch.nn.functional.relu(key_layer(normed_after_attn))

interesting_keys = [309, 381, 449,  12,  77, 197, 300, 176, 463, 487, 414]

for i, key in enumerate(interesting_keys):
    lesioned_keys = keys.clone()
    print(f"{lesioned_keys.shape = }")
    # lesioned_keys[:,key] = 0
    lesioned_keys[:,interesting_keys[:i]] = 0
    
    lesioned_probs = output_layer(model.transformer[0].norm2(value_layer(lesioned_keys) + normed_after_attn)).softmax(dim=-1)[torch.arange(len(data)),data[:,-1]]

    plt.figure()
    plt.imshow(einops.rearrange(lesioned_probs.detach().cpu().numpy(), '(h w) -> h w', h=97), origin='lower', norm=colors.LogNorm())
    plt.colorbar()
    plt.xticks(np.arange(0,97,10))
    plt.yticks(np.arange(0,97,10))
    plt.title(f"Lesioned key {key}")
    plt.show()
    plt.close()