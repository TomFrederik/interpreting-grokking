import os 

import einops
import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from datasets import get_dataset
from model import GrokkingTransformer
from utils import load_model

model_name = "Single Layer ReLU"
_, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in range(0,1071,5)]

os.makedirs(f'{model_name}/eq_query_plots', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torch.from_numpy(get_dataset('minus', 97, './data').data).to(device)[:,:-1]

for path in tqdm(paths):
    epoch = int(path.split('/')[-1].split('-')[0].split('=')[-1])
    # if os.path.exists(f"{model_name}/eq_query_plots/epoch={epoch}.jpg"):
    #     continue
    model = GrokkingTransformer.load_from_checkpoint(path).to(device)
    model.eval()
    
    embedded_data = model.pos_encoding(model.embedding(dataset))
    qkv = model.transformer[0].self_attn.qkv_proj(embedded_data)
    qkv = einops.rearrange(qkv, 'batch seq_length (num_heads head_dim) -> batch num_heads seq_length head_dim', head_dim=3*model.transformer[0].self_attn.head_dim)
    q, k, v = qkv.chunk(3, dim=-1)
    equal_queries = q[:,:,-1]
    num_keys = k[:,:,[0,2]]
    
    
    dot_product = torch.einsum('bhn,bhtn->bht', equal_queries, num_keys)
    fig, axes = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        
        axes[i].plot(np.arange(97), dot_product[[98*j for j in range(97)],i].detach().cpu().numpy(), label=["First Num", "Second Num"])
        axes[i].set_xticks(np.arange(0,97,8))
    axes[0].legend(bbox_to_anchor=(0.7, 1.05))
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f'{model_name}/eq_query_plots/epoch={epoch}.jpg')
    plt.close()

images = []
fig = plt.figure()
for i in range(0,1071,5):
    image = mgimg.imread(f"{model_name}/eq_query_plots/epoch={i}.jpg")
    images.append([plt.imshow(image)])
plt.axis('off')
my_anim = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=2000)
my_anim.save(f'{model_name}/eq_query_animation.gif', fps=10)