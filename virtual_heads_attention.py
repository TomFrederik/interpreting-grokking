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
    
    ckpts = [os.path.join(ckpt_dir, f"epoch={epoch}-step={epoch*10+9}.ckpt") for epoch in range(0, 1095, 5)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = torch.from_numpy(get_dataset('minus', 97, './data', no_op_token=True, force_data=True).data).to(device)

    inputs = dataset[:,:-1]
    targets = dataset[:,-1]
    
    uniform_nats = np.log(97)
    print(f"{uniform_nats = }")
    
    for ckpt in tqdm(ckpts[-1:]):
        print(ckpt)
        model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
        hooked_model = commons.HookedModel(model)
    
        attentions = save_attention_patterns(hooked_model, inputs)
        # each tensor in this list has shape (num_samples, num_heads, seq_len, seq_len)
        attn0 = attentions[0]
        attn1 = attentions[1]
        composed_attentions = torch.flatten(torch.einsum('bhsi,bkij->bhksj', attn0, attn1), 1, 2)
        print(composed_attentions.shape)
        first_num = composed_attentions[:,:,-1,0].reshape(97,97,16)
        # first_num = attn1[:,:,-1,0].reshape(97,97,4)
        second_num = composed_attentions[:,:,-1,1].reshape(97,97,16)
        
        # fig, axes = plt.subplots(2,2, sharex=True, figsize=(10,10))
        # for row in range(2):
        #     for col in range(2):
        #         plot = axes[row, col].imshow(first_num[...,row*2+col].detach().cpu().numpy(), origin='lower', vmin=torch.min(first_num), vmax=torch.max(first_num))
        # fig.colorbar(plot, ax=axes)
        # # plt.suptitle(f"{title} - Epoch {epoch}")
        # plt.show()
        
        fig, axes = plt.subplots(4, 4, sharex=True, figsize=(10,10))
        for row in range(4):
            for col in range(4):
                plot = axes[row, col].imshow(first_num[...,row*4+col].detach().cpu().numpy(), origin='lower', vmin=torch.min(first_num), vmax=torch.max(first_num))
        fig.colorbar(plot, ax=axes)
        # plt.suptitle(f"{title} - Epoch {epoch}")
        plt.show()
        
        fig, axes = plt.subplots(4, 4, sharex=True, figsize=(10,10))
        for row in range(4):
            for col in range(4):
                plot = axes[row, col].imshow(second_num[...,row*4+col].detach().cpu().numpy(), origin='lower', vmin=torch.min(second_num), vmax=torch.max(second_num))
        fig.colorbar(plot, ax=axes)
        plt.show()
        break
        # plt.savefig(f'{model_name}/attention_heatmaps/{folder}/epoch={epoch}.jpg')
        # plt.close()