import itertools as it

import numpy as np
import einops 
import unseal
import unseal.hooks as hooks
import torch
from model import GrokkingTransformer
from utils import load_model

ckpt, ckpt_dir = load_model()

model = GrokkingTransformer.load_from_checkpoint(ckpt)
model.eval()


data = dict()
for layer in range(2):
    data[layer] = dict()
    
    q, k, val = model.transformer[layer].self_attn.qkv_proj.weight.chunk(3)
    print(q.shape)
    o = model.transformer[layer].self_attn.o_proj.weight

    o_heads = torch.stack(o.chunk(4), dim=1)
    q_heads = torch.stack(q.chunk(4), dim=0)
    k_heads = torch.stack(k.chunk(4), dim=0)
    val_heads = torch.stack(val.chunk(4), dim=0)
    print(o_heads.shape)
    print(val_heads.shape)

    w_ov_heads = torch.einsum('ibj,bik->bjk', o_heads, val_heads)
    w_qk_heads = torch.einsum('bji,bjk->bik', q_heads, k_heads)
    
    data[layer]['ov'] = w_ov_heads.detach().numpy()
    data[layer]['qk'] = w_qk_heads.detach().numpy()

for h1, h2 in it.product(range(4), range(4)):
    ov = data[0]['ov'][h1]
    qk = data[1]['qk'][h2]
    ov2 = data[1]['ov'][h2]
    qkov = np.matmul(qk, ov)
    qktov = np.matmul(qk.T, ov)
    ovov = np.matmul(ov2, ov)
    k_score = np.sqrt(np.sum(qkov**2)) / (np.sqrt(np.sum(qk**2)) * np.sqrt(np.sum(ov**2)))
    q_score = np.sqrt(np.sum(qktov**2)) / (np.sqrt(np.sum(qk.T**2)) * np.sqrt(np.sum(ov**2)))
    v_score = np.sqrt(np.sum(ovov**2)) / (np.sqrt(np.sum(ov2**2)) * np.sqrt(np.sum(ov**2)))
    print(k_score, q_score, v_score)