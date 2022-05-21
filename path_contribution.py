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

def compute_direct_path(model: GrokkingTransformer):
    embed = model.embedding.weight
    unembed = model.output.weight.T
    return embed @ unembed

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
    
def create_attention_overwrite_hooks(num_layers: int, attentions: List[Tensor]):
    attn_overwrite_hooks = [
        commons.Hook(f'transformer->{i}->self_attn->attn', common_hooks.replace_activation("...", attentions[i]), f'fix_attn_{i}')
        for i in range(num_layers)
    ]
    return attn_overwrite_hooks

def create_ov_save_hooks(num_layers: int):
    ov_save_hooks = [
        commons.Hook(f'transformer->{i}->self_attn', common_hooks.save_output(), f'save_ov_{i}')
        for i in range(num_layers-1)
    ]
    return ov_save_hooks


def create_ov_overwrite_hooks(num_layers: int, replacement_tensors: List[Tensor]):
    ov_overwrite_hooks = [
        commons.Hook(f'transformer->{i}->self_attn', common_hooks.replace_activation("...", replacement_tensors[i], 0), f'zero_ov_{i}')
        for i in range(num_layers-1)
    ]
    return ov_overwrite_hooks

def create_stable_hooks(num_layers: int, attentions: List[Tensor]):
    attn_overwrite_hooks = create_attention_overwrite_hooks(num_layers, attentions)
    ov_save_hooks = create_ov_save_hooks(num_layers)
    return attn_overwrite_hooks, ov_save_hooks

def my_function(hooked_model: commons.HookedModel, inputs: Tensor, targets: Tensor) -> Tensor:
    
    attentions = save_attention_patterns(hooked_model, inputs)
    # each tensor in this list has shape (num_samples, num_heads, seq_len, seq_len)
    
    # run model, fix attention weights, record outputs and overwrite outputs to zero
    zero_out = torch.zeros(hooked_model.model.hparams.width).to(hooked_model.device)
    attn_overwrite_hooks, ov_save_hooks = create_stable_hooks(hooked_model.model.hparams.layers, attentions)
    ov_overwrite_hooks = create_ov_overwrite_hooks(hooked_model.model.hparams.layers, [zero_out] * (hooked_model.model.hparams.layers-1))
    
    
    first_order_out = hooked_model(inputs, hooks=attn_overwrite_hooks + ov_save_hooks + ov_overwrite_hooks)
    save_ov_0 = hooked_model.save_ctx['save_ov_0']['output'][0]
    # print((save_ov_0.to(hooked_model.device) @ hooked_model.model.output.weight.T)[:,-1])
    first_order_pred = torch.log_softmax(save_ov_0.to(hooked_model.device) @ hooked_model.model.output.weight.T + first_order_out, dim=-1)[torch.arange(len(targets)), -1, targets]
    first_order_mean = -torch.mean(first_order_pred)
    print(f"{first_order_mean = :.4f}")
    
    
    ov_overwrite_hooks = [
        commons.Hook('transformer->0->self_attn', common_hooks.replace_activation("...", save_ov_0, 0), 'zero_ov_0'),
    ]
    
    second_order_pred = torch.log_softmax(hooked_model(inputs, hooks=attn_overwrite_hooks + ov_save_hooks + ov_overwrite_hooks)[:,-1],dim=-1)
    second_order_pred = second_order_pred[torch.arange(len(targets)), targets]
    second_order_mean = -torch.mean(second_order_pred)
    print(f"{second_order_mean = :.4f}")


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
        
        my_function(hooked_model, inputs, targets)
        break
        
        direct_path_logits = compute_direct_path(model)[-1,:-1] # only look at equality token, and only to number tokens
        direct_logprobs = torch.log(torch.softmax(direct_path_logits, dim=-1))
        direct_nats = -direct_logprobs[targets].mean()
        direct_marginal_loss_red = uniform_nats - direct_nats
        print(f"Direct path: {direct_marginal_loss_red.item():.4f} nats")
        
        # compute first order path first layer
        first_order_first_layer_hook = commons.Hook('transformer->0->self_attn', common_hooks.save_output(), 'first_layer_path')
        first_order_second_layer_hook = commons.Hook('transformer->1->self_attn', common_hooks.save_output(), 'second_layer_path')
        hooked_model(inputs, hooks=[first_order_first_layer_hook, first_order_second_layer_hook])
        
        attn_outs = hooked_model.save_ctx['first_layer_path']['output'][0]
        first_order_first_layer_logits = (attn_outs[:,-1] @ model.embedding.weight.T.cpu()) # NO -> need to add initial embedding + pos
        first_order_first_layer_logprobs = torch.log(torch.softmax(first_order_first_layer_logits, dim=-1))
        first_order_first_layer_nats = -first_order_first_layer_logprobs[torch.arange(len(targets)),targets].mean()
        first_order_first_layer_loss_red = direct_nats - first_order_first_layer_nats
        
        attn_outs = hooked_model.save_ctx['second_layer_path']['output'][0]
        first_order_second_layer_logits = (attn_outs[:,-1] @ model.embedding.weight.T.cpu()) # NO -> need to add initial embedding + pos, but also need to separate out different circuits 
        first_order_second_layer_logprobs = torch.log(torch.softmax(first_order_second_layer_logits, dim=-1))
        first_order_second_layer_nats = -first_order_second_layer_logprobs[torch.arange(len(targets)),targets].mean()
        first_order_second_layer_loss_red = first_order_first_layer_nats - first_order_second_layer_nats
        print("First order:")
        print(f"First Layer: {first_order_first_layer_loss_red.item():.4f} nats")
        print(f"Second Layer: {first_order_second_layer_loss_red.item():.4f} nats")
