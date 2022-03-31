import logging
import os 

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch.nn.functional as F
import torch
from tqdm import tqdm

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

import time

def act_steps(baseline_act, num_steps):
    for i in range(num_steps):
        yield baseline_act * i/num_steps

def get_neuron_generator(baseline_acts, num_steps):
    for i in range(baseline_acts.shape[2]):
        yield act_steps(baseline_acts[:,-1,i], num_steps)

def run_model_with_new_activations(model, x, new_acts):
    with torch.no_grad():
        residual_stream = model.transformer[0].norm1(model.transformer[0].self_attn(x, mask=model.self_attn_mask[:context_len,:context_len], return_attention=True)[0] + x)
    linear_out = model.transformer[0].linear_net[-1](new_acts)
    output = model.output(model.transformer[0].norm2(linear_out + residual_stream))
    return output

@ray.remote(num_cpus=2, num_gpus=0)
def compute_model_kn_rankings(ckpt, epoch, x, y, save_dir=None, device='cpu'):
    print(f'\nStarted epoch {epoch}')
    if save_dir is None:
        save_dir = './'
    model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
    model.eval()

    # compute baseline activations of intermediate neurons
    embedded_x = model.pos_encoding(model.embedding(x.to(device))).cpu()
    baseline_acts = F.relu(
        model.transformer[0].linear_net[0](
            model.transformer[0].norm1(
                model.transformer[0].self_attn(embedded_x.to(device), mask=model.self_attn_mask[:context_len,:context_len], return_attention=True)[0] + embedded_x.to(device)
            )
        )
    )

    grads = []
    for j, sample in enumerate(tqdm(embedded_x)):
        grads.append([])
        neuron_generator = get_neuron_generator(baseline_acts, num_steps=20)
        for i, neuron in enumerate(neuron_generator):
            grad = 0
            for act in neuron:
                full_acts = baseline_acts[j].clone()
                full_acts[-1,i] = act[j]
                sample = sample.to(device)
                full_acts = full_acts.to(device)
                model.zero_grad()
                output = F.softmax(run_model_with_new_activations(model, sample[None], full_acts), dim=-1)
                loss = output[:,-1,y[j].to(device)]
                full_acts.retain_grad()
                loss.backward(retain_graph=True)
                grad += full_acts.grad[-1,i]
                full_acts.grad.data = torch.zeros_like(full_acts.grad.data)
            grads[-1].append((grad * baseline_acts[j,-1,i]/20).to('cpu'))

    np.save(os.path.join(save_dir, f"{epoch}.npy"), torch.argsort(torch.stack([torch.stack(g) for g in grads]), descending=True, dim=1).numpy())


# ray.init(log_to_driver=False)
ray.init(log_to_driver=True, num_cpus=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')

model_name = 'Single Layer ReLU'
_, ckpt_dir = load_model(model_name)

save_dir = os.path.join(model_name, 'kn_rankings')
os.makedirs(save_dir, exist_ok=True)

epochs = [294, 303, 315, 328, 340, 353, 364, 376, 389, 405, 420, 434, 457, 477, 496, 516, 532, 553, 574, 606, 648, 678, 716, 764, 800] # first batch
epochs += [307, 321, 332, 343, 356, 369, 382, 394, 411, 424, 447, 467, 486, 505, 522, 540, 561, 596, 638, 667, 703, 743, 787] # second batch

ckpts = [os.path.join(ckpt_dir, f'epoch={epoch}-step={epoch*10+9}.ckpt') for epoch in epochs]
device = 'cpu'
data_idcs = [98*i for i in range(97)]

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long()
data = data[data_idcs]
x = data[:,:-1]
y = data[:,-1]
context_len = x.shape[1]

refs = []
for epoch, ckpt in zip(epochs, ckpts):
    refs.append(compute_model_kn_rankings.remote(ckpt, epoch, x, y, save_dir=save_dir, device=device))

for epoch, ref in tqdm(zip(epochs, refs)):
    ray.get(ref)
    print(f'\nFinished epoch {epoch}')