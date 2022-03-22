import numpy as np
import plotly.express as px
import streamlit as st
import pytorch_lightning as pl
from datasets import get_dataset
import matplotlib.pyplot as plt

seed = 42
train_ratio = 0.5
pl.seed_everything(seed)

data = get_dataset(descr='minus', num_elements=97, data_dir='./data', force_data=False).data
data = data[:,[0,2,4]]
idcs = np.random.permutation(np.arange(len(data)))
train_idcs = idcs[:int(train_ratio * len(idcs))]
val_idcs = idcs[int(train_ratio * len(idcs)):]
train_set = data[train_idcs]
val_set = data[val_idcs]

train_outputs = train_set[:,-1]
val_outputs = val_set[:,-1]

train_first_num = train_set[:,0]
val_first_num = val_set[:,0]

train_second_num = train_set[:,1]
val_second_num = val_set[:,1]

plt.figure()
# plt.hist(train_outputs, bins=97, alpha=0.5, label='train')
# plt.hist(val_outputs, bins=97, alpha=0.5, label='val')
# plt.hist(train_first_num, bins=97, alpha=0.5, label='train')
# plt.hist(val_first_num, bins=97, alpha=0.5, label='val')
plt.hist(train_second_num, bins=97, alpha=0.5, label='train')
plt.hist(val_second_num, bins=97, alpha=0.5, label='val')
plt.show()

