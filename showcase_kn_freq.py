import os

import numpy as np
from tqdm import tqdm
import streamlit as st
import pandas as pd
import plotly.express as px

RELEVANT = sorted([267, 20, 113, 96, 220, 204, 219] + [476, 268, 11, 459, 403, 222])
COL_NAMES = [str(x) for x in range(512)]

def compute_mean_rankings():
    # TODO: inc into a function
    folder = "Single Layer ReLU/kn_rankings"
    files_in_folder = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files_in_folder.sort()
    epochs = [int(f.split('.')[0]) for f in files_in_folder]
    attribution_rankings = {epoch: np.load(os.path.join(folder, f)) for f, epoch in zip(files_in_folder, epochs)}
    
    for k in range(1, 21):
        # compute the frequency of each kn in the top 10 
        kn_freq = dict()
        for epoch, rankings in tqdm(attribution_rankings.items()):
            kn_freq[epoch] = [np.isin(rankings[:,:k], i, assume_unique=True).sum() / 97 for i in range(512)]
        
        # make into a dataframe
        data = np.array([val for val in kn_freq.values()])
        st.session_state['df'] = pd.DataFrame(data=data, columns=COL_NAMES)
        st.session_state['epochs'] = list(kn_freq.keys())


def generate_plots():
    for k in range(1, 21):
        fig = px.line(st.session_state.df, x=st.session_state.epochs, y=[COL_NAMES[x] for x in st.session_state.active_neurons], markers=True)
        fig.update_layout(title=f'Top-{k} Frequency', xaxis_title='Epoch', yaxis_title='Frequency')
        # for peak in peaks:
        #     plt.axvline(x=peak, color='r', linestyle='--', linewidth=1)
        st.session_state.plots.append(fig)

def init_streamlit():
    if 'init' not in st.session_state:
        compute_mean_rankings()
        st.session_state['plots'] = []
        st.session_state['active_neurons'] = RELEVANT #TODO
        generate_plots()
        st.session_state['init'] = True

# set streamlit page layout to wide
st.set_page_config(layout='wide')

# obsolete?
# peaks = [297, 309, 322, 334, 345, 358, 370, 384, 396, 414, 425, 449, 468, 487, 507, 524, 542, 563, 597, 640, 668, 704, 744, 788,]

init_streamlit()

st.slider('k', 1, 20, key='k', value=1)
st.plotly_chart(st.session_state['plots'][st.session_state['k']-1])
