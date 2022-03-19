import streamlit as st
from typing import Optional, Union, Iterable, Callable

from unseal.hooks import Hook
from unseal.hooks.common_hooks import transformers_get_attention
from unseal.hooks.util import create_slice_from_str
from unseal.visuals.streamlit_interfaces import SESSION_STATE_VARIABLES, interface_setup, utils


# perform startup tasks
interface_setup.startup(SESSION_STATE_VARIABLES, './registered_models.json')

# create sidebar
with st.sidebar:
    interface_setup.create_sidebar()
    
    sample = st.checkbox('Enable sampling', value=False, key='sample')
    if sample:
        interface_setup.create_sample_sliders()
        interface_setup.on_sampling_config_change()
    
    if "storage" not in st.session_state:
        st.session_state["storage"] = ["", ""]
        
    # input 1
    placeholder1 = st.empty()
    placeholder1.text_area(label='Input 1', on_change=utils.on_text_change, key='input_text_1', value=st.session_state.storage[0], kwargs=dict(col_idx=0, text_key='input_text_1'))
    if sample:
        st.button(label="Sample", on_click=utils.sample_text, kwargs=dict(col_idx=0, key="input_text_1"), key="sample_text_1")
    
    # input 2
    placeholder2 = st.empty()
    placeholder2.text_area(label='Input 2', on_change=utils.on_text_change, key='input_text_2', value=st.session_state.storage[1], kwargs=dict(col_idx=1, text_key='input_text_2'))
    if sample:
        st.button(label="Sample", on_click=utils.sample_text, kwargs=dict(col_idx=1, key="input_text_2"), key="sample_text_2")
    
    # sometimes need to force a re-render
    st.button('Show Attention', on_click=utils.text_change, kwargs=dict(col_idx=[0,1]))
    
    # f =  json.encoder.JSONEncoder().encode(st.session_state.visualization)
    # st.download_button(
    #     label='Download Visualization', 
    #     data=f, 
    #     file_name=f'{st.session_state.model_name}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.json', 
    #     mime='application/json', 
    #     help='Download the visualizations as an json of html files.', 
    #     key='download_button'
    # )

# show the html visualization
if st.session_state.model is not None:
    cols = st.columns(2)
    for col_idx, col in enumerate(cols):
        if f"col_{col_idx}" in st.session_state.visualization:
            with col:
                for layer in range(st.session_state.num_layers):
                    if f"layer_{layer}" in st.session_state.visualization[f"col_{col_idx}"]:
                        with st.expander(f'Layer {layer}'):
                            st.components.v1.html(st.session_state.visualization[f"col_{col_idx}"][f"layer_{layer}"], height=600, scrolling=True)
        else:
            st.session_state.visualization[f"col_{col_idx}"] = dict()