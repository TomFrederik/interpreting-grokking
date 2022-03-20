import os

import numpy as np
import plotly.express as px
import streamlit as st

def startup():
    if 'startup' not in st.session_state:
        model_name = "Single Layer ReLU"
        equal_inputs_logits = np.load(os.path.join(f"{model_name}", 'equal_inputs_logits.npy'))
        st.session_state['all_logits'] = equal_inputs_logits 
        
        st.session_state['figure'] = None
        st.session_state['startup'] = True
        
        st.set_page_config(layout='wide')


def get_active_data():
    return st.session_state.all_logits[st.session_state.epoch_range[0]:st.session_state.epoch_range[1]+1]


def create_figure():
    active_data = get_active_data()
    return create_heatmap(active_data)

def create_heatmap(data):
    fig = px.imshow(
        data.T,
        labels = {'x': 'Epoch', 'y': 'Input', 'z': 'Logit'},
        x = list(range(st.session_state.epoch_range[0], st.session_state.epoch_range[-1]+1)),
        y=list(range(97)),
        width=1200, height=800,
        origin = 'lower',
    )
    return fig
    
def on_range_change():
    # update figure
    st.session_state.figure = create_figure()

def display_figure():
    if st.session_state.figure is None:
        st.session_state.figure = create_figure()
        
    st.plotly_chart(st.session_state.figure)
    
startup()

# title
st.header('Select epoch range')

st.slider('Epoch', 0, len(st.session_state.all_logits)-1, [394,404], step=1, key='epoch_range', on_change=on_range_change)

display_figure()


