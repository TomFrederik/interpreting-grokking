import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
from unseal.hooks import HookedModel, Hook
from unseal.hooks.common_hooks import create_attention_hook
from unseal.logit_lense import logit_hook

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')


ckpt, ckpt_dir = load_model()

model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
model = HookedModel(model)
model.eval()

output_layer = model.model.output

attention_hook = create_attention_hook(
    layer=0, 
    key='attention_hook', 
    output_idx=1,
    attn_name='self_attn',
    layer_key_prefix='transformer',
    heads='1:3',
)

num = 14
idx = num * 97

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().to(device)
x = data[idx:idx+1,:-1]
context_len = x.shape[1]

# model(x, hooks=[attention_hook])
# print(model.save_ctx['attention_hook']['attn'].shape)

# compute baseline activations of intermediate neurons
embedded_x = model.model.pos_encoding(model.model.embedding(x))

prior = model.model.output(embedded_x).softmax(dim=-1)
attn_out = model.model.transformer[0].self_attn(embedded_x)
mlp_in = model.model.transformer[0].norm1(attn_out+embedded_x)

intermediate = model.model.output(mlp_in).softmax(dim=-1)

memory_keys = F.gelu(model.model.transformer[0].linear_net[0](mlp_in))[0,-1]
k = 100
topk_keys = torch.topk(memory_keys, k=k, dim=-1)[1]

pos_keys = torch.where(memory_keys>0)[0]

value_logits = model.model.output(model.model.transformer[0].linear_net[-1].weight.T).T
centered_value_logits = value_logits - value_logits.mean(dim=0, keepdim=True)
print(centered_value_logits[num, pos_keys].sort(descending=True)[0])
print((value_logits[:,pos_keys].argsort(dim=0, descending=True) == num).nonzero(as_tuple=True)[0])

mlp_out = model.model.transformer[0].linear_net(mlp_in)

final = model.model.output(model.model.transformer[0].norm2(mlp_in + mlp_out)).softmax(dim=-1)
# plt.figure()
# plt.bar(np.arange(len(prior[0,-1])), prior[0,-1].detach().cpu())
# plt.show()

# plt.figure()
# plt.bar(np.arange(len(intermediate[0,-1])), intermediate[0,-1].detach().cpu())
# plt.xticks(np.arange(0,len(intermediate[0,-1]),10))
# plt.show()

# plt.figure()
# plt.bar(np.arange(len(final[0,-1])), final[0,-1].detach().cpu())
# plt.xticks(np.arange(0,len(final[0,-1]),10))
# plt.show()