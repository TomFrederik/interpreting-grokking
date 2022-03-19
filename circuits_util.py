import itertools
import math
from typing import Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from model import GrokkingTransformer, MultiheadAttention
from utils import load_model

def get_qkv_weights(attention_module: MultiheadAttention) -> Tuple[Tensor, Tensor, Tensor]:
    qkv = einops.rearrange(attention_module.qkv_proj.weight, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=3*attention_module.head_dim)
    q, k, v = qkv.chunk(3, dim=1)
    return q, k, v

def get_o_weight(attention_module: MultiheadAttention) -> Tensor:
    return einops.rearrange(attention_module.o_proj.weight, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)

def get_qk_circuit(attention_module: MultiheadAttention, head: int):
    q, k, _ = get_qkv_weights(attention_module)
    qk = q[head].T @ k[head]
    return qk

def get_ov_circuit(transformer: GrokkingTransformer, layer: int, head: int):
    *_, v = get_qkv_weights(transformer.transformer[layer].self_attn)
    o = get_o_weight(transformer.transformer[layer].self_attn)
    out = transformer.output.weight
    emb = transformer.embedding.weight
    ov = out @ o[head].T @ v[head] @ emb.T
    return ov

def composition(a: Tensor, b: Tensor):
    return (a.T @ b).norm(p='fro') / (a.norm(p='fro') * b.norm(p='fro'))

def q_composition(qk: Tensor, ov: Tensor):
    return composition(qk, ov)

def k_composition(qk: Tensor, ov: Tensor):
    return composition(qk.T, ov)

def v_composition(ov_2: Tensor, ov_1: Tensor):
    return composition(ov_2.T, ov_1)

def get_init_limits(weight: Tensor) -> float:
    # compute xavier uniform initialization limits
    return (-math.sqrt(6/(weight.shape[0] + weight.shape[1])), math.sqrt(6/(weight.shape[0] + weight.shape[1])))

def approx_baseline(shape_1, shape_2, limits_1, limits_2, num_samples):
    baseline = 0
    for i in range(num_samples):
        mat_1 = torch.distributions.Uniform(limits_1[0], limits_1[1]).rsample(shape_1)
        mat_2 = torch.distributions.Uniform(limits_2[0], limits_2[1]).rsample(shape_2)
        
        baseline += composition(mat_1, mat_2)
    
    return baseline / num_samples

def compute_all_baselines(attention_module: MultiheadAttention, num_samples):
    q, k, v = get_qkv_weights(attention_module)
    o = get_o_weight(attention_module)
    # print(f"q: {q.shape}")
    # print(f"k: {k.shape}")
    # print(f"v: {v.shape}")
    # print(f"o: {o.shape}")
    qk_shape = (q.shape[1],) + (k.shape[1],)
    ov_shape = (o.shape[1],) + (v.shape[1],)
    # print(f"{qk_shape = }")
    # print(f"{ov_shape = }")
    qk_limits = get_init_limits(attention_module.qkv_proj.weight)
    ov_limits = get_init_limits(o)
    
    qk_baseline = approx_baseline(qk_shape, ov_shape, qk_limits, ov_limits, num_samples)
    v_baseline = approx_baseline(ov_shape, ov_shape, ov_limits, ov_limits, num_samples)
        
    return qk_baseline, v_baseline

@torch.no_grad()
def compute_all_compositions(attn_1: MultiheadAttention, attn_2: MultiheadAttention, num_samples: int = 1000):
    qk_baseline, v_baseline = compute_all_baselines(attn_2, num_samples)
    # print(f"{qk_baseline = }")
    # print(f"{v_baseline = }")
    
    q_1, k_1, v_1 = get_qkv_weights(attn_1)
    # print(f"v_1: {v_1.shape}")
    o_1 = get_o_weight(attn_1)
    # print(f"o_1: {o_1.shape}")
    ov_1 = torch.einsum('abc, acd -> abd', einops.rearrange(o_1, 'a c b -> a b c'), v_1)
    qk_1 = torch.einsum('abc, acd -> abd', einops.rearrange(q_1, 'a c b -> a b c'), k_1)
    # print(f"{qk_1.shape = }")
    # print(f"{ov_1.shape = }")
    q_2, k_2, v_2 = get_qkv_weights(attn_2)
    o_2 = get_o_weight(attn_2)
    ov_2 = torch.einsum('abc, acd -> abd', einops.rearrange(o_2, 'a c b -> a b c'), v_2)
    qk_2 = torch.einsum('abc, acd -> abd', einops.rearrange(q_2, 'a c b -> a b c'), k_2)
    # print(f"{qk_2.shape = }")
    # print(f"{ov_2.shape = }")
    
    q_comps = []
    k_comps = []
    v_comps = []
    
    for head_1, head_2 in itertools.product(range(attn_1.num_heads), range(attn_2.num_heads)):
        q_comps.append(q_composition(qk_2[head_2], ov_1[head_1]))
        k_comps.append(k_composition(qk_2[head_2], ov_1[head_1]))
        v_comps.append(v_composition(ov_2[head_2], ov_1[head_1]))
    
    # q_comps = torch.stack(q_comps) - qk_baseline
    # k_comps = torch.stack(k_comps) - qk_baseline
    # v_comps = torch.stack(v_comps) - v_baseline
    
    q_comps = torch.stack(q_comps)
    k_comps = torch.stack(k_comps)
    v_comps = torch.stack(v_comps)
    
    
    return q_comps.clamp(min=0).numpy(), k_comps.clamp(min=0).numpy(), v_comps.clamp(min=0).numpy()


if __name__ == "__main__":
        
    ckpt, ckpt_dir = load_model()
    
    # epochs = list(range(190,201))
    epochs = list(range(0,410))
    ckpts = [ckpt_dir + f"epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in epochs]
    models = [GrokkingTransformer.load_from_checkpoint(ckpt) for ckpt in ckpts]

    q_comps = []
    k_comps = []
    v_comps = []
    for model in tqdm(models):
        q, k, v = compute_all_compositions(model.transformer[0].self_attn, model.transformer[1].self_attn)
        
        q_comps.append(q)
        k_comps.append(k)
        v_comps.append(v)

    q_comps = np.array(q_comps).reshape(-1, 4, 4)
    k_comps = np.array(k_comps).reshape(-1, 4, 4)
    v_comps = np.array(v_comps).reshape(-1, 4, 4)
    
    # q_max = q_comps.max(1).max(1)
    # k_max = k_comps.max(1).max(1)
    # v_max = v_comps.max(1).max(1)

    # plt.figure()
    # plt.plot(q_max, label='Q')
    # plt.plot(k_max, label='K')
    # plt.plot(v_max, label='V')
    # plt.legend()
    # plt.title('Max')
    # plt.show()

    for i in range(4):
        fig, axes = plt.subplots(4,1, sharex=True)
        for j in range(4):
            axes[j].plot(q_comps[:,j,i], label='Q')
            axes[j].plot(k_comps[:,j,i], label='K')
            axes[j].plot(v_comps[:,j,i], label='V')
        plt.legend()
        plt.savefig(f'head_{i}.png')

        
        