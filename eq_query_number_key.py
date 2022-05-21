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

model_name = "Attention Only"
_, ckpt_dir = load_model(model_name)
paths = [os.path.join(ckpt_dir, file) for file in os.listdir(ckpt_dir) if (os.path.isfile(os.path.join(ckpt_dir, file)) and file.endswith(".ckpt"))]
# get epochs from path
epochs = [int(path.split('/')[-1].split('-')[0].split('=')[-1]) for path in paths]

os.makedirs(f'{model_name}/eq_query_plots', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torch.from_numpy(get_dataset('minus', 97, './data', no_op_token=True, force_data=True).data).to(device)[:,:-1]

exponent = True

for i, path in enumerate(tqdm(paths)):
    epoch = int(path.split('/')[-1].split('-')[0].split('=')[-1])
    if epoch % 5 != 0:
        continue
    model = GrokkingTransformer.load_from_checkpoint(path).to(device)
    model.eval()
    
    embedded_data = model.pos_encoding(model.embedding(dataset))
    qkv = model.transformer[0].self_attn.qkv_proj(embedded_data)
    qkv = einops.rearrange(qkv, 'batch seq_length (num_heads head_dim) -> batch num_heads seq_length head_dim', head_dim=3*model.transformer[0].self_attn.head_dim)
    q, k, v = qkv.chunk(3, dim=-1)
    equal_queries = q[:,:,-1]
    
    
    dot_product = torch.einsum('bhn,bhtn->bht', equal_queries, k)[[98*j for j in range(97)]]
    dot_product -= torch.mean(dot_product, dim=-1, keepdim=True)
    dot_product /= 128**0.5
    dot_product = dot_product[...,[0,1]]
    
    if exponent:
        dot_product = torch.exp(dot_product)
    
    num_heads = dot_product.shape[1]
    fig, axes = plt.subplots(num_heads, 1, sharex=True)
    for i in range(num_heads):
        
        axes[i].plot(np.arange(97), dot_product[:,i].detach().cpu().numpy(), label=["First Num", "Second Num"])
        # axes[i].plot(np.arange(97), (dot_product[:,i,0] - dot_product[:,i,1]).detach().cpu().numpy())
        axes[i].set_xticks(np.arange(0,97,8))
    axes[0].legend(bbox_to_anchor=(0.7, 1.05))
    plt.suptitle(f"Epoch {epoch}")
    # ##########
    # plt.show()
    # raise ValueError
    # ##########
    plt.savefig(f'{model_name}/eq_query_plots/epoch={epoch}.jpg')
    plt.close()

# images = []
# fig = plt.figure()
# for i in epochs:
#     image = mgimg.imread(f"{model_name}/eq_query_plots/epoch={i}.jpg")
#     images.append([plt.imshow(image)])
# plt.axis('off')
# my_anim = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=2000)
# my_anim.save(f'{model_name}/eq_query_animation.gif', fps=10)