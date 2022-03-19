import os 

import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt
import numpy as np
import torch
# from torch.fft import rfft
from tqdm import tqdm

from circuits_util import get_qkv_weights
from model import GrokkingTransformer
from utils import load_model

model_name = "Single Layer ReLU"
_, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in range(0,929,5)]

os.makedirs(f'{model_name}/eq_query_plots', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for path in tqdm(paths):
    epoch = int(path.split('/')[-1].split('-')[0].split('=')[-1])
    # if os.path.exists(f"{model_name}/eq_query_plots/epoch={epoch}.jpg"):
    #     continue
    model = GrokkingTransformer.load_from_checkpoint(path).to(device)

    number_embedding = model.embedding.weight[:-2]
    equal_embedding = model.embedding.weight[-1]

    q, k, v = get_qkv_weights(model.transformer[0].self_attn)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
    for i in range(4):
        number_keys = torch.einsum('ij,kj->ik', k[i], number_embedding)
        eq_query = q[i] @ equal_embedding
        dot_product = eq_query @ number_keys
        
        axes[i].plot(np.arange(len(dot_product)), dot_product.detach().cpu().numpy())
        axes[i].set_xticks(np.arange(0,97,8))
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f'{model_name}/eq_query_plots/epoch={epoch}.jpg')
    plt.close()

images = []
fig = plt.figure()
for i in range(0,929,5):
    image = mgimg.imread(f"{model_name}/eq_query_plots/epoch={i}.jpg")
    images.append([plt.imshow(image)])
plt.axis('off')
my_anim = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=2000)
my_anim.save(f'{model_name}/eq_query_animation.gif', fps=10)