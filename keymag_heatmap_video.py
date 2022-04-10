
import logging
import os

import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from unseal.hooks import Hook, HookedModel
from unseal.hooks.common_hooks import save_output

from datasets import get_dataset
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')
    
model_name = 'Single Layer ReLU'
ckpt, ckpt_dir = load_model(model_name)

img_dir = os.path.join(model_name, 'keymag_images')
os.makedirs(img_dir, exist_ok=True)

# model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)

epochs = list(range(0,1050,5))
ckpts = [os.path.join(ckpt_dir, f"epoch={epoch}-step={epoch*10+9}.ckpt") for epoch in epochs]
dataset = torch.from_numpy(get_dataset(descr='minus', num_elements=97, data_dir='./data').data).to(device)

for epoch, ckpt in tqdm(zip(epochs, ckpts)):
    # if the file already exists skip it
    if os.path.exists(os.path.join(img_dir, f"{epoch}.png")):
        continue
    
    model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
    model.eval()

    key_layer = model.transformer[0].linear_net[0].weight
    value_layer = model.transformer[0].linear_net[-1].weight

    neurons = [204, 219, 222, 403]

    # What's the distribution over key magnitudes for these three value vectors, over the whole dataset?
    # Which inputs activate them the most?
    hooked_model = HookedModel(model)
    save_key_mag_hook = Hook("transformer->0->linear_net->2", save_output(), key="save_key_magnitude")
    hooked_model(dataset[:,:-1], hooks=[save_key_mag_hook])

    key_magnitude = hooked_model.save_ctx['save_key_magnitude']['output'][:,-1,neurons]

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, gridspec_kw={'hspace':0.2, 'wspace':-0.4})
    for i, neuron in enumerate(neurons):
        row = i // 2
        col = i % 2
        axes[row,col].imshow(key_magnitude[:,i].reshape(97,97).detach().cpu().numpy(), origin='lower', cmap='viridis')
        axes[row,col].set_title(f'{neuron}')
    # plt.colorbar()
    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(os.path.join(img_dir, f"{epoch}.png"))
    plt.close()


# now generate video
images = []
fig = plt.figure()
for i in range(0,1050,5):
    image = mgimg.imread(f"{model_name}/keymag_images/{i}.png")
    images.append([plt.imshow(image)])
plt.axis('off')
my_anim = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=2000)
my_anim.save(f'{model_name}/keymag_heatmap_animation.gif', fps=10)
