import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

range2suffix = {'0-1200':'', '400-800':'_400_800', '500-600':'_500_600'}


def load_data():
    model_name = "Single Layer ReLU"
    pca_results = np.load(os.path.join(f"{model_name}", f'functionspace_pca_results{range2suffix[st.session_state.epoch_range]}.npz'))
    st.session_state['transformed_data'] = pca_results['transformed_data']
    st.session_state["explained_variance_ratio"] = pca_results['explained_variance_ratio']
    
def startup():
    if 'startup' not in st.session_state: # only execute this once
        st.session_state['epoch_range'] = '0-1200'
        load_data()
        st.session_state['active'] = []
        st.session_state['num_active_components'] = 0
        st.session_state['total_explained_variance'] = 0
        st.session_state['figure'] = None
        st.session_state['startup'] = True
        st.session_state['lines'] = False
        
        st.set_page_config(layout='wide')

def update_active_components(i):
    try:
        idx = st.session_state.active.index(i) # in list
        del st.session_state.active[idx]
    except ValueError: # not in list
        st.session_state.active.append(i)
        # check if list if full
        if len(st.session_state.active) > 3:
            del st.session_state.active[0]
    # sort list
    st.session_state.active.sort()
    
    st.session_state.num_active_components = get_num_active_components()
    st.session_state.total_explained_variance = get_total_explained_variance()
    
def get_num_active_components():
    return len(st.session_state.active)

def get_total_explained_variance():
    return sum(st.session_state.explained_variance_ratio[st.session_state.active])


def get_active_data():
    return st.session_state.transformed_data[:,st.session_state.active]


def create_figure():
    u = get_active_data()
    if st.session_state.num_active_components == 2:
        fig = create_2d_figure(u, st.session_state.active)
    elif st.session_state.num_active_components == 3:
        fig = create_3d_figure(u, st.session_state.active)
    else:
        fig = None
    return fig

def create_3d_figure(data, active_components):
    labels = [f'{i+1}. PC' for i in sorted(active_components)]
    
    if st.session_state.lines:
        fig = go.Figure(
            data=go.Scatter3d(
                x=data[:,0], 
                y=data[:,1], 
                z=data[:,2], 
                marker=dict(
                    size=2,
                    color=np.arange(len(data)),
                ),
                line=dict(
                    color='grey',
                    width=1
                )
            ),
            layout=go.Layout(
                title=go.layout.Title(text=f'Explained Variance: {st.session_state.total_explained_variance:.2f}'),
                width=1200, 
                height=800,
            )
        )
        fig.update_layout(
            scene = dict(
                xaxis_title=labels[0],
                yaxis_title=labels[1],
                zaxis_title=labels[2],
            ),
        )
    else:
        fig = px.scatter_3d(
        data_frame = {labels[0]: data[:,0], labels[1]: data[:,1], labels[2]: data[:,2], 'Epoch':np.arange(len(data))}, 
        x=labels[0], 
        y=labels[1], 
        z=labels[2], 
        color='Epoch', 
        title=f'Explained Variance: {st.session_state.total_explained_variance:.2f}',
        width=1200, height=800,
    )
    fig.update_layout(
        scene=dict(
            yaxis = dict(
                showticklabels=False,
            ), 
            xaxis = dict(
                showticklabels=False,
            ),
            zaxis = dict(
                showticklabels=False,
            ),
        ),
        coloraxis_colorbar = dict(
            tickvals=np.arange(0,len(data),100),
        )
    )
    fig.update_traces(marker=dict(size=2))
    return fig

def create_2d_figure(data, active_components):
    labels = [f'{i+1}. PC' for i in sorted(active_components)]
    
    if st.session_state.lines:
        fig = go.Figure(
            data=go.Scatter(
                x=data[:,0], 
                y=data[:,1], 
                marker=dict(
                    size=5,
                    color=np.arange(len(data)),
                ),
                line=dict(
                    color='grey',
                    width=1
                ),
                mode='markers+lines' if st.session_state.lines else 'markers',
            ),
            layout=go.Layout(
                title=go.layout.Title(text=f'Explained Variance: {st.session_state.total_explained_variance:.2f}'),
                width=1200, 
                height=800,
            ),
        )
        fig.update_layout(
            scene = dict(
                xaxis_title=labels[0],
                yaxis_title=labels[1],
            ),
        )
    else:
        fig = px.scatter(
            data_frame = {labels[0]: data[:,0], labels[1]: data[:,1], 'Epoch':np.arange(len(data))}, 
            x=labels[0], 
            y=labels[1], 
            color='Epoch', 
            title=f'Explained Variance: {st.session_state.total_explained_variance:.2f}',
            width=1200, height=800,
        )
    fig.update_layout(
        yaxis_showticklabels=False, 
        xaxis_showticklabels=False, 
        coloraxis_colorbar=dict(
            tickvals=np.arange(0,len(data),100),
        )
    )

    return fig
    
def update_figure():
    st.session_state.figure = create_figure()    
    
def on_checkbox_change(i):
    # update actives
    update_active_components(i)
    
    # update figure
    update_figure()

def display_figure():
    if st.session_state.figure is not None:
        st.plotly_chart(st.session_state.figure)

def update_range():
    # delete all active components
    st.session_state.active = []
    st.session_state['num_active_components'] = 0
    st.session_state['total_explained_variance'] = 0
    st.session_state['figure'] = None
    load_data()

startup()

with st.sidebar:

    with st.form('Config'):
        
        st.selectbox('Range', options=["0-1200", "400-800",'500-600'], index=0, key='epoch_range')
        
        st.form_submit_button('Update range!', on_click=update_range)


    st.checkbox('Show lines', value=False, key='lines', on_change=update_figure)
    
    # title
    st.header('Select Principal Components (up to 3)')
    
    for i in range(len(st.session_state["explained_variance_ratio"])):
        st.checkbox(f"{i+1:02d}. ({st.session_state.explained_variance_ratio[i]:.3f})", value=i in st.session_state.active, key=f"component_{i}", on_change=on_checkbox_change, args=[i])

    st.write(f"Total Expl. Variance: {st.session_state.total_explained_variance:.4f}")


display_figure()


