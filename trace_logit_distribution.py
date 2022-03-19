import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')

model_name = 'Single Layer ReLU'
ckpt, ckpt_dir = load_model(model_name)

model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
model.eval()

key_layer = model.transformer[0].linear_net[0]
value_layer = model.transformer[0].linear_net[-1]
output_layer = model.output

gamma1 = model.transformer[0].norm1.weight
beta1 = model.transformer[0].norm1.bias
gamma2 = model.transformer[0].norm2.weight
beta2 = model.transformer[0].norm2.bias
# print(f'{gamma.shape} {beta.shape}') #[128], [128]

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().to(device)
x = data[:,:-1]
# pass data through the 
print(model.embedding(x))
x = model.pos_encoding(model.embedding(x))
print(x)
context_len = x.shape[1]
attn_out, _ = model.transformer[0].self_attn(x, model.self_attn_mask[:context_len,:context_len], return_attention=True)

var1 = (attn_out + x).var(unbiased=False, dim=-1, keepdim=True)
mean1 = (attn_out + x).mean(dim=-1, keepdim=True)
scaling_factor1 = gamma1 / torch.sqrt(var1 + 1e-5)
summand1 = mean1 * scaling_factor1

normed_after_attn = model.transformer[0].norm1(x + attn_out)
linear_out = model.transformer[0].linear_net(normed_after_attn)
var2 = (linear_out + normed_after_attn).var(unbiased=False, dim=-1, keepdim=True) # used in layernorm
mean2 = (linear_out + normed_after_attn).mean(dim=-1, keepdim=True)
scaling_factor2 = gamma2 / torch.sqrt(var2 + 1e-5)

# think this is a bit non-sensical because the scaling factor is not a constant but differs for different dimensions
rescaled_prior = output_layer(scaling_factor1 * scaling_factor2 * x)
rescaled_attn_logits = output_layer(scaling_factor1 * scaling_factor2 * attn_out - scaling_factor2 * summand1)
mlp_logits = output_layer(scaling_factor2 * linear_out)
beta1_logits = output_layer(beta1 * scaling_factor2)
beta2_logits = output_layer(beta2)

print(x.mean())
num = 14
idx = num * 97
print((scaling_factor2 * summand1)[idx,-1])
print((scaling_factor1 * scaling_factor2 * attn_out)[idx,-1])

sign_agreement = rescaled_attn_logits[idx,-1,:-2] * mlp_logits[idx,-1,:-2]
colors = list(map(lambda x: 'r' if x < 0 else 'g', sign_agreement))

print(f"{data[idx] = }")
print('\nSanity check that everything adds up again:')
print((beta2_logits + beta1_logits[idx,-1] + rescaled_prior[idx,-1] + mlp_logits[idx,-1] + rescaled_attn_logits[idx,-1]).argmax())
print('')

fig, axes = plt.subplots(4,1,sharex=True, gridspec_kw={'hspace': 0.5})
fig.suptitle('14 - 0 = 14')
axes[0].bar(np.arange(97), rescaled_prior[idx,-1,:-2].detach().cpu().numpy().flatten())
axes[0].set_title('Rescaled Prior Logits')
# axes[1].bar(np.arange(97), rescaled_attn_logits[idx,-1,:-2].detach().cpu().numpy().flatten())
# axes[1].set_title('Rescaled Attention Logits')
axes[1].bar(np.arange(97), (rescaled_prior[idx,-1,:-2] + rescaled_attn_logits[idx,-1,:-2]).detach().cpu().numpy().flatten())
axes[1].set_title('Rescaled Attention+Prior Logits')
axes[2].bar(np.arange(97), mlp_logits[idx,-1,:-2].detach().cpu().numpy().flatten(), color=colors)
axes[2].set_title('MLP Logits')
axes[3].bar(np.arange(97), (rescaled_prior[idx,-1,:-2] + mlp_logits[idx,-1,:-2] + rescaled_attn_logits[idx,-1,:-2]).detach().cpu().numpy().flatten())
axes[3].set_title('Result Logits')
plt.xticks(np.arange(0,97,8))
plt.show()

