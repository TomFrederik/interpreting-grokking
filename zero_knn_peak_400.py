
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')

model_name = 'Single Layer ReLU'
ckpt, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in [395,400]]
models = [GrokkingTransformer.load_from_checkpoint(ckpt).to(device) for ckpt in paths]

inputs = [
    0,2,9,11,12,14,15,16,17,19,21,23,26,27,31,34,36,37,47,48,
    49,50,52,54,56,58,60,62,63,65,66,67,68,70,71,75,76,78,79,
    80,81,83,85,86,87,88,89,92,93,95,96
]
idcs = [i*98 for i in inputs]

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().to(device)
x = data[idcs,:-1]
y = data[idcs,-1]
context_len = x.shape[1]

model_grads = []
for model in tqdm(models, desc="Model"):
    # compute baseline activations of intermediate neurons
    embedded_x = model.pos_encoding(model.embedding(x))
    baseline_acts = F.relu(model.transformer[0].linear_net[0](model.transformer[0].norm1(model.transformer[0].self_attn(embedded_x, mask=model.self_attn_mask[:context_len,:context_len], return_attention=True)[0] + embedded_x)))
    grads = []
    for j, sample in enumerate(tqdm(embedded_x, desc='Sample', leave=False)):
        grads.append([])
        neuron_generator = get_neuron_generator(baseline_acts, num_steps=20)
        for i, neuron in enumerate(tqdm(neuron_generator, leave=False, desc='Neuron')):
            grad = 0
            for act in neuron:
                full_acts = baseline_acts[j].clone()
                full_acts[-1,i] = act[j]
                model.zero_grad()
                output = F.softmax(run_model_with_new_activations(model, sample[None], full_acts), dim=-1)
                loss = output[:,-1, 0]
                full_acts.retain_grad()
                loss.backward(retain_graph=True)
                grad += full_acts.grad[-1,i]
                full_acts.grad.data = torch.zeros_like(full_acts.grad.data)
            grads[-1].append(grad * baseline_acts[j,-1,i]/20)
    grads = torch.stack([torch.stack(g) for g in grads])
    print(grads.shape)
    model_grads.append(grads)

model_grads = torch.stack(model_grads)
print(model_grads)
print(model_grads.shape)
np.save(f"{model_name}/knn_zero_grads_395_400.npy", model_grads.detach().cpu().numpy())


