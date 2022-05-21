from typing import List, Union
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch

from unseal.hooks import Hook, HookedModel
from unseal.hooks.common_hooks import replace_activation

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

def lesion_head(head_idx: Union[int, List] = 3, head_dim=32):
    if isinstance(head_idx, int):
        indices = f":,:,{head_idx * head_dim}:{(head_idx + 1) * head_dim}"
        hook_fn = replace_activation(indices=indices, replacement_tensor=torch.zeros(head_dim), tuple_index=0)
        hook = Hook('transformer->0->self_attn', hook_fn, f'lesion_head_{head_idx}')
    elif isinstance(head_idx, list):
        print(head_idx)
        idx_list = [head_dim*i+j for i in head_idx for j in range(head_dim)]
        print(idx_list)
        indices = f":,:,{idx_list}"
        hook_fn = replace_activation(indices=indices, replacement_tensor=torch.zeros(head_dim*len(head_idx)), tuple_index=0)
        hook = Hook('transformer->0->self_attn', hook_fn, f'lesion_head_{head_idx}')
    else:
        raise TypeError
    return hook

model_name = "No Norm, Single Layer"
ckpt, ckpt_dir = load_model(model_name)
model = GrokkingTransformer.load_from_checkpoint(ckpt).cuda()
model = HookedModel(model)


data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().cuda()
y = data[:,-1]
y_hat = model(data[:,:-1], [])
probs = torch.softmax(y_hat, dim=1)
probs = probs[torch.arange(len(data)), -1, y]

reverse_lesion = False

for i in range(4):
    if reverse_lesion:
        hook = lesion_head([j for j in range(4) if j != i])
    else:
        hook = lesion_head(i)

    y_hat_lesioned = model.forward(data[:,:-1], [hook])
    lesioned_probs = torch.softmax(y_hat_lesioned, dim=1)
    lesioned_probs = lesioned_probs[torch.arange(len(data)), -1, y].detach().cpu().numpy()

    plt.imshow(einops.rearrange(lesioned_probs, '(h w) -> h w', h=97), cmap='viridis', origin='lower')
    plt.xticks(np.arange(0,97,8))
    plt.yticks(np.arange(0,97,8))
    plt.colorbar()
    plt.title(f'Head {i}')
    if reverse_lesion:
        plt.savefig(f'{model_name}/reverse_lesion_head_{i}.jpg')
    else:
        plt.savefig(f'{model_name}/lesion_head_{i}.jpg')
    plt.close()

