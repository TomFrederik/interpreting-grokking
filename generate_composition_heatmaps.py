import os

import numpy as np
from tqdm import tqdm

from circuits_util import compute_all_baselines, compute_all_compositions
from model import GrokkingTransformer
import matplotlib.pyplot as plt
from utils import load_model, get_epoch


def main():

    model_name = "Attention Only"
    os.makedirs(f'{model_name}/composition_heatmaps', exist_ok=True)
    
    _, ckpt_dir = load_model(model_name)
    ckpts = sorted([os.path.join(ckpt_dir, file) for file in os.listdir(ckpt_dir) if file.endswith(".ckpt")], key=lambda x: get_epoch(x))
    q_comps = []
    k_comps = []
    v_comps = []
    baselines = None
    for ckpt in tqdm(ckpts):
        model = GrokkingTransformer.load_from_checkpoint(ckpt)
        if baselines is None:
            baselines = compute_all_baselines(model.transformer[0].self_attn, num_samples=1000)
        
        q, k, v = compute_all_compositions(model.transformer[0].self_attn, model.transformer[1].self_attn, baselines)
        q_comps.append(q)
        k_comps.append(k)
        v_comps.append(v)

    q_comps = np.array(q_comps).reshape(-1, 4, 4)
    k_comps = np.array(k_comps).reshape(-1, 4, 4)
    v_comps = np.array(v_comps).reshape(-1, 4, 4)
    # each row corresponds to a head in layer 0
    # each column corresponds to a head in layer 1

    for i in tqdm(range(q_comps.shape[0])):
        epoch = get_epoch(ckpts[i])

        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        fig.suptitle(f'Epoch {epoch}')

        axes[0].set_title('Q')
        axes[0].imshow(q_comps[i], cmap='hot', vmin=0, vmax=0.1, interpolation='nearest', origin='lower')
        axes[0].set_xlabel('Layer 1')
        axes[0].set_ylabel('Layer 0')
        
        axes[1].set_title('K')
        axes[1].get_yaxis().set_visible(False)
        axes[1].imshow(k_comps[i], cmap='hot', vmin=0, vmax=0.1, interpolation='nearest', origin='lower')
        axes[1].set_xlabel('Layer 1')

        axes[2].set_title('V')
        axes[2].get_yaxis().set_visible(False)
        axes[2].set_xlabel('Layer 1')
        plot = axes[2].imshow(v_comps[i], cmap='hot', vmin=0, vmax=0.1, interpolation='nearest', origin='lower')

        fig.colorbar(plot, ax=axes)
        plt.savefig(f'{model_name}/composition_heatmaps/epoch_{epoch}.jpg')
        plt.close()
        
if __name__ == "__main__":
    main()
