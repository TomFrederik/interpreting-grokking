import einops
import matplotlib.pyplot as plt
import numpy as np
import torch

from unseal.hooks import Hook, HookedModel
from unseal.hooks.common_hooks import replace_activation

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

def lesion_head(head_idx=3, head_dim=32):
    indices = f":,:,{head_idx * head_dim}:{(head_idx + 1) * head_dim}"
    hook_fn = replace_activation(indices=indices, replacement_tensor=torch.zeros(head_dim), tuple_index=0)
    hook = Hook('transformer->0->self_attn', hook_fn, f'lesion_head_{head_idx}')
    return hook

model_name = "Single Layer ReLU"
ckpt, ckpt_dir = load_model(model_name)
model = GrokkingTransformer.load_from_checkpoint(ckpt).cuda()
model = HookedModel(model)


data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().cuda()
y = data[:,-1]
y_hat = model(data[:,:-1], [])
probs = torch.softmax(y_hat, dim=1)
probs = probs[torch.arange(len(data)), -1, y]


for i in range(4):
    hook = lesion_head(i)

    y_hat_lesioned = model.forward(data[:,:-1], [hook])
    lesioned_probs = torch.softmax(y_hat_lesioned, dim=1)
    lesioned_probs = lesioned_probs[torch.arange(len(data)), -1, y].detach().cpu().numpy()

    plt.imshow(einops.rearrange(lesioned_probs, '(h w) -> h w', h=97), cmap='viridis', origin='lower')
    plt.xticks(np.arange(0,97,8))
    plt.yticks(np.arange(0,97,8))
    plt.colorbar()
    plt.title(f'Head {i}')
    plt.savefig(f'{model_name}/lesion_head_{i}.jpg')
    plt.close()

