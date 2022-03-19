import os 

import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt
import numpy as np
import torch
# from torch.fft import rfft
from tqdm import tqdm

from circuits_util import get_qkv_weights, get_qk_circuit
from model import GrokkingTransformer
from utils import load_model

model_name = "Single Layer ReLU"
_, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in [395,400]]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model1 = GrokkingTransformer.load_from_checkpoint(paths[0]).to(device)
model2 = GrokkingTransformer.load_from_checkpoint(paths[1]).to(device)
number_embedding1 = model1.embedding.weight[:-2]
equal_embedding1 = model1.embedding.weight[-1]
number_embedding2 = model2.embedding.weight[:-2]
equal_embedding2 = model2.embedding.weight[-1]

q1, k1, v1 = get_qkv_weights(model1.transformer[0].self_attn)
q2, k2, v2 = get_qkv_weights(model2.transformer[0].self_attn)
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)

for i in range(4):
    number_keys1 = torch.einsum('ij,kj->ik', k1[i], number_embedding1)
    eq_query1 = q1[i] @ equal_embedding1
    dot_product1 = eq_query1 @ number_keys1

    number_keys2 = torch.einsum('ij,kj->ik', k2[i], number_embedding2)
    eq_query2 = q2[i] @ equal_embedding2
    dot_product2 = eq_query2 @ number_keys2
    
    axes[i].plot(np.arange(len(dot_product1))[:2], (dot_product2-dot_product1)[[47,48]].detach().cpu().numpy())
    # axes[i].set_xticks(np.arange(0,97,8))
    axes[i].set_xticks([0,1],[47,48])

plt.suptitle(f"Diff 400-395")
plt.show()
raise ValueError
plt.savefig(f'{model_name}/eq_query_diff_395_400.jpg')
plt.close()