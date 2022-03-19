import logging
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from unseal.hooks import HookedModel, Hook

from model import GrokkingTransformer, GrokkingTokenizer
from utils import load_model

def logit_hook_attn(layer, out_module, *args, **kwargs):
    def inner(save_ctx, input, output):
        module_in = output[0]
        save_ctx['logits'] = out_module(module_in).detach().cpu()
    
    # write key
    key = str(layer) + '_logits_attn'
    
    # create hook
    hook = Hook(f'transformer->{layer}->norm1', inner, key)
    
    return hook


def logit_hook_mlp(layer, out_module, gamma, beta):
    def inner(save_ctx, input, output):
        module_in = output[0]
        
        # scaled = gamma * (module_in - module_in.mean(dim=-1, keepdim=True)) / ((module_in + input[0]).var(dim=-1, keepdim=True) + 1e-5).sqrt() + beta
        scaled = gamma * (input[0] - input[0].mean(dim=-1, keepdim=True)) / ((module_in + input[0]).var(dim=-1, keepdim=True) + 1e-5).sqrt()
        # scaled = module_in + input[0]
        # scaled = gamma * (scaled - scaled.mean(dim=-1, keepdim=True)) / (scaled.var(unbiased=False, dim=-1, keepdim=True) + 1e-5).sqrt() + beta
        
        # save_ctx['logits'] = out_module(module_in).detach().cpu()
        save_ctx['logits'] = out_module(scaled).detach().cpu()
    
    # write key
    key = str(layer) + '_logits_mlp'
    
    # create hook
    # hook = Hook(f'transformer->{layer}->norm2', inner, key)
    hook = Hook(f'transformer->{layer}->linear_net', inner, key)
    
    return hook


def generate_logit_lense(
    model: HookedModel, 
    tokenizer, 
    sentence: str,
    layers: Optional[List[int]] = None,
    ranks: Optional[bool] = False,
    kl_div: Optional[bool] =False,
    include_input: Optional[bool] =False,
):
    # prepare model input
    tokenized_sentence = tokenizer.encode(sentence, return_tensors='pt').to(model.device)
    targets = tokenizer.encode(sentence)[1:]
    
    # instantiate hooks
    num_layers = 1
    if layers is None:
        layers = list(range(num_layers))
    logit_hooks = [logit_hook(layer, model.model.output, gamma=model.model.transformer[layer].norm2.weight, beta=model.model.transformer[layer].norm2.bias) for layer in layers for logit_hook in [logit_hook_attn, logit_hook_mlp]]
    
    # run model
    model.forward(tokenized_sentence, hooks=logit_hooks)
    logits = torch.stack([model.save_ctx[str(layer) + f'_logits_{layer_type}']['logits'] for layer in range(num_layers) for layer_type in ['attn', 'mlp']], dim=0)
    if include_input:
        logits = torch.cat([model.model.output(model.model.pos_encoding(model.model.embedding(tokenized_sentence))), logits], dim=0)
    logits = F.log_softmax(logits, dim=-1)
    
    # compute ranks and kld
    if ranks:
        inverted_ranks = torch.argsort(logits, dim=-1, descending=True)
        ranks = torch.argsort(inverted_ranks, dim=-1) + 1
        ranks = ranks[:, torch.arange(len(targets)), targets]
    else:
        ranks = None

    if kl_div: # Note: logits are already normalized internally by the logit_hook
        kl_div = F.kl_div(logits, logits[-1][None], reduction='none', log_target=True).sum(dim=-1)
        kl_div = kl_div[:, torch.arange(len(targets)), targets]
    else:
        kl_div = None    
        
    logits = logits[:, torch.arange(len(targets)), targets]
    
    return logits, ranks, kl_div


ckpt, ckpt_dir = load_model()

model = GrokkingTransformer.load_from_checkpoint(ckpt)

model = HookedModel(model)
sentence = '14 - 0 = 14'
tokenizer = GrokkingTokenizer()

logits, ranks, kl_div = generate_logit_lense(model, tokenizer, sentence, include_input=True)

plt.figure(figsize=(10,10))
plt.imshow(logits.detach().cpu().numpy(), cmap='hot', origin='lower')
plt.yticks([0,1,2], ['input', 'attn_1', 'mlp_1'])
plt.xticks([0,1,2,3], ['14', '-', '0', '='])
plt.colorbar()
plt.title('Logit Lense')
plt.show()

