
import logging
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from unseal.hooks import Hook, HookedModel
from unseal.hooks.common_hooks import save_output
import numpy as np
# import scipy.stats as stats

from datasets import get_dataset
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')
    
model_name = 'No Norm, Single Layer'
ckpt, ckpt_dir = load_model(model_name)

img_dir = os.path.join(model_name)
os.makedirs(img_dir, exist_ok=True)

# model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)

epochs = list(range(0,2000,5))
ckpts = [os.path.join(ckpt_dir, f"epoch={epoch}-step={epoch*10+9}.ckpt") for epoch in epochs]
dataset = torch.from_numpy(get_dataset(descr='minus', num_elements=97, data_dir='./data').data).to(device)
num_non_zeros = []
# skews = []
# neurons = [204, 219, 222, 403]
for epoch, ckpt in tqdm(zip(epochs, ckpts)):
    model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
    model.eval()

    # What's the distribution over key magnitudes for these three value vectors, over the whole dataset?
    # Which inputs activate them the most?
    hooked_model = HookedModel(model)
    save_key_mag_hook = Hook("transformer->0->linear_net->2", save_output(), key="save_key_magnitude")
    hooked_model(dataset[:,:-1], hooks=[save_key_mag_hook])

    key_magnitude = hooked_model.save_ctx['save_key_magnitude']['output'][:,-1,list(range(512))]
    num_non_zeros.append([len(key_magnitude[key_magnitude[:,i] != 0, i])/(97**2) for i in range(512)])
    # skews.append([stats.skew(key_magnitude[key_magnitude[:,i] != 0,i]) for i in range(512)])

num_non_zeros = np.array(num_non_zeros)
# skew = np.array(skews)

plt.figure()
plt.plot(epochs, np.median(num_non_zeros, axis=1), 'r--',label='median')
plt.plot(epochs, np.mean(num_non_zeros, axis=1), 'k--', label=r'$\mu\pm1\sigma$')
plt.fill_between(epochs, np.mean(num_non_zeros, axis=1) - np.std(num_non_zeros, axis=1), np.mean(num_non_zeros, axis=1) + np.std(num_non_zeros, axis=1), color='k', alpha=0.2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Share Non-zero")
plt.title("Non-zero key magnitudes")
plt.savefig(os.path.join(img_dir, "non_zero_key_mags_mean_median.png"))


# plt.figure()
# plt.plot(epochs, np.median(skews, axis=1), 'r--',label='median')
# plt.plot(epochs, np.mean(skews, axis=1), 'k--', label=r'$\mu\pm1\sigma$')
# plt.fill_between(epochs, np.mean(skews, axis=1) - np.std(skews, axis=1), np.mean(skews, axis=1) + np.std(skews, axis=1), color='k', alpha=0.2)
# plt.legend()
# plt.xlabel("Epoch")
# plt.ylabel("Skewness")
# plt.title("Skew of key magnitudes")
# plt.savefig(os.path.join(img_dir, "skew_key_mags_mean_median.png"))



# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, gridspec_kw={'hspace':0.3, 'wspace':0.1})
# for i, neuron in enumerate(neurons):
#     axes[i//2, i%2].plot(epochs, num_non_zeros[:,neuron])
#     axes[i//2, i%2].plot(epochs, np.median(num_non_zeros, axis=1), 'r--')
#     axes[i//2, i%2].plot(epochs, np.mean(num_non_zeros, axis=1), 'k--')
#     axes[i//2, i%2].fill_between(epochs, np.mean(num_non_zeros, axis=1) - np.std(num_non_zeros, axis=1), np.mean(num_non_zeros, axis=1) + np.std(num_non_zeros, axis=1), color='k', alpha=0.2)
#     axes[i//2, i%2].set_title(f"{neurons[i]}")
#     if i//2 == 1:
#         axes[i//2, i%2].set_xlabel("Epoch")
#     if i % 2 == 0:
#         axes[i//2, i%2].set_ylabel("Share Non-zero")
# plt.suptitle("Non-zero key magnitudes")
# plt.savefig(os.path.join(img_dir, "non_zero_key_mags.png"))