import os

import einops
import numpy as np
import torch

from datasets import get_dataset

total_logit_matrix = torch.load(os.path.join('Single Layer ReLU', 'total_logit_matrix.pt'))

print(f'{total_logit_matrix.shape = }') # should be (N, 97^2 * 99)

total_logit_matrix = einops.rearrange(total_logit_matrix, 'steps (num_samples outputs) -> steps num_samples outputs', num_samples = 97**2, outputs = 99)
total_logit_matrix = torch.nn.functional.log_softmax(total_logit_matrix, dim=-1)
print(f'{total_logit_matrix.shape = }') # should be (N, 97^2, 99)

sample_idcs = [98*i for i in range(97)] # correspond to "x - x ="
label_idcs = [0] * 97 # answer is always 0

# verify that this indexing is correct
dataset = get_dataset(descr='minus', num_elements=97, data_dir='./data', force_data=False).data
print(f'{dataset[sample_idcs][:10] = }')

equal_inputs = total_logit_matrix[:, sample_idcs, label_idcs]
print(f'{equal_inputs.shape = }') # should be (N, 97)

np.save(os.path.join('Single Layer ReLU', 'equal_inputs_logits.npy'), equal_inputs.numpy())

