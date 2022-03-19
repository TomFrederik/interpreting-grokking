import logging
import os

import einops
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.image as mgimg
import torch.nn.functional as F
import torch
from tqdm import tqdm

from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')


model_name = "Single Layer ReLU"
ckpt, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in range(0, 1108, 5)]

os.makedirs(f'{model_name}/dot_embedding_plots', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch, path in enumerate(tqdm(paths[-1:])):
    # if os.path.exists(f"dot_embedding_plots/ep|och={epoch}.jpg"):
    #     continue
    model = GrokkingTransformer.load_from_checkpoint(path).to(device)

    embedding = model.embedding.weight
    reference = embedding[-1] # compare to equal sign
    
    dot_product = (reference[None] * embedding).sum(dim=1) / torch.norm(reference, p=2) / torch.norm(embedding, dim=1, p=2)
    
    chunked_embedding = einops.rearrange(embedding, 'b (num_heads head_dim) -> b num_heads head_dim', num_heads = 4)
    assert chunked_embedding.shape == (embedding.shape[0], 4, 32), chunked_embedding.shape
    chunked_reference = einops.rearrange(reference, '(num_heads head_dim) -> num_heads head_dim', num_heads = 4)
    print(chunked_reference.shape)
    unscaled_chunked_dot_product = (chunked_reference * chunked_embedding).sum(dim=-1)
    
    # fig, axes = plt.subplots(4,1,sharex=True,sharey=True)
    # for i in range(4):  
    #     axes[i].plot(np.arange(0,97,1), dot_products[:,i])
    # plt.suptitle(f'Epoch {epoch}')
    
    plt.figure()
    # plt.plot(np.arange(0,99,1), dot_product.detach().cpu().numpy())
    fig, axes = plt.subplots(4,1,sharex=True,sharey=True)
    for i in range(4):
        axes[i].plot(np.arange(0,99,1), unscaled_chunked_dot_product[:,i].detach().cpu().numpy())
    # plt.plot(np.arange(0,99,1), dot_product.detach().cpu().numpy())
    # plt.title(f'Epoch {epoch}')
    plt.show()
    # plt.savefig(f'dot_embedding_plots/epoch={epoch}.jpg')
    plt.close()


images = []
fig = plt.figure()
for i in range(0,1108,5):
    image = mgimg.imread(f"{model_name}/dot_embedding_plots/epoch={i}.jpg")
    images.append([plt.imshow(image)])
plt.axis('off')
my_anim = animation.ArtistAnimation(fig, images, interval=200, repeat_delay=2000)
my_anim.save(f'{model_name}/dot_embedding_plots.gif', fps=30)