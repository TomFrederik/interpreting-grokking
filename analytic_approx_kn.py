import numpy as np
import torch
from scipy.stats import spearmanr

from datasets import get_dataset
from model import GrokkingTransformer
from utils import load_model

model_name = "Single Layer ReLU"
ckpt, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in [395,400]]

# test with a single model
model = GrokkingTransformer.load_from_checkpoint(paths[0]).to('cuda')

unembed = model.output.weight
mlp_weights = model.transformer[0].linear_net[-1].weight

dataset = torch.from_numpy(get_dataset(descr='minus', num_elements=97, data_dir='./data', force_data=False).data).to('cuda')
x = dataset[96*98:96*98+1,:-1]
y = dataset[96*98:96*98+1,-1]

x = model.pos_encoding(model.embedding(x))
x_pre_mlp = model.transformer[0].norm1(model.transformer[0].self_attn(x) + x)[:,-1]
x = model.transformer[0].linear_net[:2](x_pre_mlp) # apply first linear layer and relu
x = torch.nn.functional.relu(x)

unmodified_out =  x @ mlp_weights.T + x_pre_mlp
mean = unmodified_out.mean(dim=1, keepdim=True)
var = unmodified_out.var(dim=1, keepdim=True)
gamma = model.transformer[0].norm2.weight

# look at a single neuron
EPS = 1e-5
neuron = 0
grads = []
print(f"{x.shape = }")
for i in range(512):
    grad = (mlp_weights[:,i:i+1].T * gamma / torch.sqrt(var + EPS)) @ unembed.T * x[:,i]
    grads.append(grad[0,0].item())

exact_grads = np.load(f"{model_name}/knn_zero_grads_395_400.npy")

ranking1 = np.argsort(grads)[::-1]
ranking2 = np.argsort(exact_grads[-1,0])[::-1]
# compute correlation between attribution score rankings before and after peak
rankings = np.concatenate([ranking1[None], ranking2[None]], axis=0)
corr = spearmanr(rankings, axis=1)[0]
print(f"{corr = }")
