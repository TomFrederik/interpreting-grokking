import itertools

import einops
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GrokkingTransformer
from utils import load_model
import datasets

@st.cache(allow_output_mutation=True)
def startup(top_k):
    model_name = 'Single Layer ReLU'
    ckpt, ckpt_dir = load_model(model_name)

    epochs = list(range(394,405))
    ckpts = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in epochs]

    data = torch.from_numpy(datasets.get_dataset(descr='minus', num_elements=97, data_dir='data').data).to('cuda')
    
    input = data[:, :-1]
    input_pairs = data[:, [0,2]]
    labels = data[:, -1]

    logit_list = []
    for ckpt in tqdm(ckpts):
        model = GrokkingTransformer.load_from_checkpoint(ckpt).to('cuda')
        output = F.log_softmax(model(input)[:,-1], dim=-1)
        logits = output[torch.arange(len(output)), labels]
        logit_list.append(logits.detach().cpu())
    # compute difference
    # top movers
    top_k_logits = torch.argsort((logit_list[-1] - logit_list[0]), dim=-1, descending=True)[:top_k]
    # top peakers
    # top_k_logits = torch.argsort((logit_list[0] - logit_list[4]), dim=-1, descending=True)[:top_k]
    top_k_logit_list = [l[top_k_logits] for l in logit_list]
    top_k_input_pairs = input_pairs[top_k_logits].detach().cpu()
    top_k_labels = labels[top_k_logits].detach().cpu()
    
    figures = plotting(top_k_input_pairs, top_k_labels, top_k_logit_list) 
    return input_pairs, labels, logit_list, figures
    
def plotting(input_pairs, labels, logits):
    figures = []
    for i in range(len(logits)):
        img_data = logits[i].numpy()
        img_data = einops.rearrange(img_data, '(x y) -> x y', x=5, y=10)
        fig = plt.figure(i, figsize=(10,5))
        plt.title(f"x - y mod 97 = z")
        plt.imshow(img_data, cmap='RdBu', vmin=-10, vmax=0)
        plt.colorbar(label='Logits')
        for j in range(len(input_pairs)):
            plt.text(j // 5, j % 5, f"({input_pairs[j][0]},{input_pairs[j][1]}, {labels[j]})", fontsize=8, va='center', ha='center')  
        figures.append(fig)
    return figures

top_k = 50

input_pairs, labels, logit_list, figs = startup(top_k)
logits = torch.stack(logit_list, dim=0)

st.slider(label='Epoch', min_value=394, max_value=404, value=394, key='epoch')

st.pyplot(figs[st.session_state.epoch - 394])
# logit_by_label = torch.stack([logits[:, labels==i] for i in range(97)], dim=0)[:,4]
logit_by_label = torch.stack([logits[:, labels==i] for i in range(97)], dim=0)[:,:].mean(dim=-1)

col1, col2 = st.columns(2)
with col1:
    st.slider(label='z', min_value=0, max_value=96, value=0, key='z')

with col2:
    fig = plt.figure(figsize=(30,30))
    plt.imshow(logit_by_label.numpy(), cmap='RdBu', vmin=-10, vmax=0)
    plt.colorbar(label='Logits')
    st.pyplot(fig)