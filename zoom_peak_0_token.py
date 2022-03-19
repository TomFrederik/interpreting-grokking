import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GrokkingTransformer
from utils import load_model
import datasets

model_name = 'Single Layer ReLU'
ckpt, ckpt_dir = load_model(model_name)

epochs = list(range(395,401))
ckpts = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in epochs]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = torch.from_numpy(datasets.get_dataset(descr='minus', num_elements=97, data_dir='data').data).to(device)

idcs = [98*i for i in range(97) if i in [47,48]]
data = dataset[idcs]
x = data[:,:-1]
y = data[:,-1]

probs = []
for i, epoch in enumerate(epochs):
    model = GrokkingTransformer.load_from_checkpoint(ckpts[i]).to(device)
    probs.append(model(x)[:,-1].softmax(dim=-1)[torch.arange(len(x)), y])

probs = torch.stack(probs, dim=0).T

plt.figure()
plt.imshow(probs.detach().cpu().numpy(), origin='lower', cmap='RdBu')
plt.colorbar(label='p(0)')
plt.xlabel('epoch')
plt.show()

