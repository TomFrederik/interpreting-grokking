import os

import numpy as np
from tqdm import tqdm
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
RELEVANT = sorted([267, 20, 113, 96, 220, 204, 219] + [476, 268, 11, 459, 403, 222])
COL_NAMES = [str(x) for x in range(512)]

def compute_rel_scores():
    # TODO: inc into a function
    folder = "Single Layer ReLU/kn_attr_scores"
    files_in_folder = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files_in_folder.sort()
    epochs = sorted([int(f.split('.')[0]) for f in files_in_folder])
    attribution_scores = {epoch: np.load(os.path.join(folder, f)) for f, epoch in zip(files_in_folder, epochs)}
    data = []
    kn_scores = dict()
    for epoch, scores in tqdm(attribution_scores.items()):
        kn_scores[epoch] = scores 
    
    # make into a dataframe
    data = np.array([val for val in kn_scores.values()])
    
    st.session_state['data'] = data
    st.session_state['epochs'] = list(kn_scores.keys())


def generate_plots():
    st.session_state['plots'] = []

    for i in range(11):
        fig = plt.figure()
        plt.hist(st.session_state.data[i,:,st.session_state.neuron], bins=50)

        st.session_state.plots.append(fig)

def init_streamlit():
    if 'init' not in st.session_state:
        compute_rel_scores()
        for i in COL_NAMES:
            if i not in st.session_state:
                st.session_state[i] = False
        st.session_state['init'] = True
    
    generate_plots()

# set streamlit page layout to wide
# st.set_page_config(layout='wide')

# obsolete?
# peaks = [297, 309, 322, 334, 345, 358, 370, 384, 396, 414, 425, 449, 468, 487, 507, 524, 542, 563, 597, 640, 668, 704, 744, 788,]

with st.sidebar:
    
    st.markdown("# Neurons")
    st.selectbox("Select Neuron", RELEVANT, key='neuron')


init_streamlit()
st.slider('Epoch', 0, 10, key='epoch')
st.pyplot(st.session_state.plots[st.session_state.epoch])
