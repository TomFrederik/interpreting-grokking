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

ckpt, ckpt_dir = load_model()

model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
model.eval()

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().to(device)
x = data[:,:-1]
y = data[:,-1]
context_len = x.shape[1]

# compute baseline activations of intermediate neurons
embedded_x = model.pos_encoding(model.embedding(x))
baseline_acts = F.gelu(model.transformer[0].linear_net[0](model.transformer[0].norm1(model.transformer[0].self_attn(embedded_x, mask=model.self_attn_mask[:context_len,:context_len], return_attention=True)[0] + embedded_x)))

grads = []
for j, sample in enumerate(tqdm(embedded_x)):
    grads.append([])
    neuron_generator = get_neuron_generator(baseline_acts, num_steps=20)
    for i, neuron in enumerate(neuron_generator):
        grad = 0
        for act in neuron:
            full_acts = baseline_acts[j].clone()
            full_acts[-1,i] = act[j]
            model.zero_grad()
            output = F.softmax(run_model_with_new_activations(model, sample[None], full_acts), dim=-1)
            loss = output[:,-1,y[j]]
            full_acts.retain_grad()
            loss.backward(retain_graph=True)
            grad += full_acts.grad[-1,i]
            full_acts.grad.data = torch.zeros_like(full_acts.grad.data)
        grads[-1].append((grad * baseline_acts[j,-1,i]/20).to('cpu'))

    np.save("knn_rankings.npy", torch.argsort(torch.stack([torch.stack(g) for g in grads]), descending=True, dim=1).numpy())