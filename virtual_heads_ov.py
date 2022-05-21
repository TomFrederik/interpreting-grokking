import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import GrokkingTransformer
from utils import load_model
from datasets import get_dataset
from tqdm import tqdm
import unseal
from unseal.hooks import common_hooks, commons
from circuits_util import get_o_weight, get_qkv_weights

from typing import TypeVar, List

Tensor = TypeVar('Tensor', bound=torch.Tensor)

def create_attention_save_hooks(num_layers: int):
    attn_save_hooks = [
        common_hooks.create_attention_hook(layer=i, key=f'attn_{i}', layer_key_prefix="transformer", output_idx=1, attn_name='self_attn')
        for i in range(num_layers)
    ]
    return attn_save_hooks

def save_attention_patterns(hooked_model: commons.HookedModel, inputs: Tensor) -> List[Tensor]:
    # run model normally, recording attention weights
    attn_save_hooks = create_attention_save_hooks(hooked_model.model.hparams.layers)
    hooked_model(inputs, hooks=attn_save_hooks)
    return [hooked_model.save_ctx[f'attn_{i}']['attn'] for i in range(hooked_model.model.hparams.layers)]
    

    

if __name__ == "__main__":
        
    model_name = "Attention Only"
    ckpt, ckpt_dir = load_model(model_name)
    
    model: GrokkingTransformer = GrokkingTransformer.load_from_checkpoint(ckpt)
    embedding = model.embedding.weight
    print(embedding.shape)
    
    layer_0_o = get_o_weight(model.transformer[0].self_attn)
    layer_1_o = get_o_weight(model.transformer[1].self_attn)
    _, _, layer_0_v = get_qkv_weights(model.transformer[0].self_attn)
    _, _, layer_1_v = get_qkv_weights(model.transformer[1].self_attn)

    layer_0_h_3 = layer_0_o[3].T @ layer_0_v[3]
    layer_1_h_0 = layer_1_o[0].T @ layer_1_v[0]
    # layer_1_h_0 = torch.eye(256)
    # layer_0_h_3 = torch.eye(256)
    print(layer_0_h_3.shape)
    print(layer_1_h_0.shape)
    
    virt_ov = (embedding @ layer_1_h_0 @ layer_0_h_3 @ embedding.T).cpu().detach().numpy()
    print(virt_ov.shape)
    
    plt.imshow(virt_ov, origin='lower')
    plt.colorbar()
    plt.show()
    
    
    # fig, axes = plt.subplots(4, 4, sharex=True, figsize=(10,10))
    # for row in range(4):
    #     for col in range(4):
    #         plot = axes[row, col].imshow(first_num[...,row*4+col].detach().cpu().numpy(), origin='lower', vmin=torch.min(first_num), vmax=torch.max(first_num))
    # fig.colorbar(plot, ax=axes)
    # # plt.suptitle(f"{title} - Epoch {epoch}")
    # plt.show()
    