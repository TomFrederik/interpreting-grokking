import plotly.express as px
import streamlit as st
import torch
import numpy as np
import itertools as it

model_name = "Single Layer ReLU"
total_logit_matrix = torch.from_numpy(np.load(f'{model_name}/total_logit_matrix.npy'))


(u, s, v) = torch.pca_lowrank(total_logit_matrix[:1200], q=2, niter=2)
u = u.numpy() # this is the projected matrix
colors = np.linspace(0, 1, len(u))

x_range = np.arange(97)
y_range = np.arange(97)
xy_range = it.product(x_range, y_range)
z = np.array(list(map(lambda xy: (xy[0] - xy[1]) % 97, xy_range)))

fig = px.scatter({'x':u[:,0], 'y':u[:,1], 'color':colors}, x='x', y='y', color='color', hover_data={
    'epoch':np.arange(len(u)),
    'x':False,
    'y':False,
    'color':False,
    })

# Plot!
st.plotly_chart(fig, use_container_width=True)