import os 

import einops
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from datasets import get_dataset
from model import GrokkingTransformer
from utils import load_model, get_epoch

model_name = "Attention Only"
# model_name = "Many Heads"
_, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in range(0, 1095, 5) if os.path.exists(ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt")]
# get epochs from path
epochs = [int(path.split('/')[-1].split('-')[0].split('=')[-1]) for path in paths]

os.makedirs(f'{model_name}/attention_heatmaps/second_num', exist_ok=True)
os.makedirs(f'{model_name}/attention_heatmaps/first_num', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torch.from_numpy(get_dataset('minus', 97, './data', no_op_token=True, force_data=True).data).to(device)[:,:-1]

softmax = True
if not softmax:
    raise ValueError("This code is not ready for non-softmax attention")

for first_num in [True, False]:
    for path in tqdm(paths):
        epoch = get_epoch(path)
        
        model = GrokkingTransformer.load_from_checkpoint(path).to(device)
        model.eval()
        
        embedded_data = model.pos_encoding(model.embedding(dataset))
        qkv = model.transformer[0].self_attn.qkv_proj(embedded_data)
        qkv = einops.rearrange(qkv, 'batch seq_length (num_heads head_dim) -> batch num_heads seq_length head_dim', head_dim=3*model.transformer[0].self_attn.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        equal_queries = q[:,:,-1]
        
        pos = 0 if first_num else 1
        title = 'First Num' if first_num else 'Second Num'
        folder = 'first_num' if first_num else 'second_num'
        
        dot_product = torch.einsum('bhn,bhtn->bht', equal_queries, k)
        dot_product -= torch.mean(dot_product, dim=-1, keepdim=True)
        dot_product /= 128**0.5
        if softmax: 
            dot_product = torch.softmax(dot_product, dim=-1)
        dot_product = einops.rearrange(dot_product, '(h w) heads seq -> h w heads seq', h=97, w=97)[...,[0,1]]
        num_heads = 4
        fig, axes = plt.subplots(2, 2, sharex=True)
        for row in range(2):
            for col in range(2):
                plot = axes[row, col].imshow(dot_product[...,row*2+col,pos].detach().cpu().numpy(), origin='lower', vmin=0, vmax=1)
        fig.colorbar(plot, ax=axes)
        plt.suptitle(f"{title} - Epoch {epoch}")
        plt.savefig(f'{model_name}/attention_heatmaps/{folder}/epoch={epoch}.jpg')
        plt.close()