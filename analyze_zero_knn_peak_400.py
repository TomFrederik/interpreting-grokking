import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import rbo
from scipy.stats import spearmanr
from tqdm import tqdm
import streamlit as st
import pandas as pd
import plotly.express as px



def compute_overlap(ranking1, ranking2):
    assert len(ranking1) == len(ranking2)
    
    cur_overlap = 0
    overlaps = []
    seen = set()
    for k in range(len(ranking1)):
        if k > 10:
            break # TODO
        seen.add(ranking2[k])
        if ranking1[k] in seen:
            cur_overlap += 1
        overlaps.append(cur_overlap/(k+1))
    return overlaps        

def compute_all_overlaps(rankings):
    
    overlaps = []
    for i in range(rankings.shape[0]-1):
        for j in range(i+1, rankings.shape[0]):
            overlapAtK = compute_overlap(rankings[i], rankings[j])
            overlaps.append((i, j, overlapAtK))
    return overlaps
    
def compute_mean_topk_overlap(rankings, k=10):
    """
    Compute the mean top-k overlap between all rankings.
    """
    num_rankings = rankings.shape[0]
    mean_topk_overlap = 0
    for i in range(num_rankings-1):
        for j in range(i+1, num_rankings):
            # mean_topk_overlap += np.isin(rankings[i,:k], rankings[j,:k])
            mean_topk_overlap += len(np.intersect1d(rankings[i,:k], rankings[j,:k]))
    
    return 2 * mean_topk_overlap / (num_rankings * (num_rankings-1) * k)

def compute_mean_rbo(rankings, p=1):
    num_rankings = rankings.shape[0]
    result = 0
    for i in range(num_rankings-1):
        for j in range(i+1, num_rankings):
            result += rbo.RankingSimilarity(rankings[i], rankings[j]).rbo(p=p)
    return 2 * result / ((num_rankings-2) * (num_rankings-1))

def compute_topk_rbo(rankings, p=1, k=10):
    num_rankings = rankings.shape[0]
    results = []
    for i in range(num_rankings-1):
        for j in range(i+1, num_rankings):
            results.append(rbo.RankingSimilarity(rankings[i], rankings[j]).rbo(p=p))
    return np.mean(sorted(results)[-k:])





folder = "Single Layer ReLU/kn_rankings"

files_in_folder = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
files_in_folder.sort()
epochs = [int(f.split('.')[0]) for f in files_in_folder]
attribution_rankings = {epoch: np.load(os.path.join(folder, f)) for f, epoch in zip(files_in_folder, epochs)}
# print(f'{epochs =}')
inputs = [
    0,2,9,11,12,14,15,16,17,19,21,23,26,27,31,34,36,37,47,48,
    49,50,52,54,56,58,60,62,63,65,66,67,68,70,71,75,76,78,79,
    80,81,83,85,86,87,88,89,92,93,95,96
]

use_good = False

peaks = [297, 309, 322, 334, 345, 358, 370, 384, 396, 414, 425, 449, 468, 487, 507, 524, 542, 563, 597, 640, 668, 704, 744, 788,]

mode = 'kn_freq'

if mode == 'overlap':
    overlaps = dict()
    for epoch, ranking in tqdm(attribution_rankings.items()):    
        # overlaps[epoch] = [compute_mean_topk_overlap(ranking, k) for k in range(1, 512)]
        
        overlaps[epoch] = compute_all_overlaps(ranking)
        # print(sorted(overlaps[epoch], key=lambda x: x[2])[-10:])
        
        # overlaps[epoch] = [compute_mean_topk_overlap(ranking, k) for k in range(1, 30)]

    
    plt.figure()
    x = np.array(list(overlaps.keys()))
    y = [np.sort([v[-1][-1] for v in val])[-10:].mean() for val in overlaps.values()]
    plt.scatter(x, y)
    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--', linewidth=1)
    # for epoch, overlap in overlaps.items():
    #     plt.plot(np.arange(len(overlap)), overlap, label=f'{epoch}')
    # plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Overlap')
    plt.title('Mean Top-20 Overlap')
    plt.savefig('Single Layer ReLU/mean_top20_overlap.png')
    
elif mode == 'rbo':

    p = 0.8

    rbo_per_epoch = dict()
    for epoch, rankings in tqdm(attribution_rankings.items()):
        # rbo_per_epoch[epoch] = compute_topk_rbo(rankings, p=p)
        rbo_per_epoch[epoch] = compute_mean_rbo(rankings, p=p)

    

    plt.figure()
    # for epoch, overlap in overlaps.items():
    #     plt.plot(np.arange(len(overlap)), overlap, label=f"{epoch}")
    plt.scatter(list(rbo_per_epoch.keys()), list(rbo_per_epoch.values()))
    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--', linewidth=1)
    # plt.plot(list(rbo_per_epoch.keys()), list(rbo_per_epoch.values()))
    plt.title(r'Mean RBO ($p=$'+ f'{p})')
    plt.xlabel('Epoch')
    plt.ylabel('RBO')
    plt.savefig('Single Layer ReLU/kn_mean_rbo.jpg')


elif mode == 'kn_freq':
    
    if 'init' not in st.session_state:
        st.session_state['plots'] = []
        for k in range(1, 21):
            # compute the frequency of each kn in the top 10 
            kn_freq = dict()
            for epoch, rankings in tqdm(attribution_rankings.items()):
                kn_freq[epoch] = [np.isin(rankings[:,:k], i, assume_unique=True).sum() / 97 for i in range(512)]

            fig = plt.figure()
            relevant_neurons = [267, 20, 113, 96, 220, 204, 219] + [476, 268, 11, 459, 403, 222]
            col_names = [str(x) for x in relevant_neurons]
            # if i in [267, 220, 219, 96, 204]:
            #     lw = 3
            # else:
            #     lw = 1
                
                # construct a dataframe
            data = np.array([val for val in kn_freq.values()])
            df = pd.DataFrame(data=data[:,relevant_neurons], columns=col_names)

            fig = px.line(df, x=list(kn_freq.keys()), y=col_names, markers=True)
            fig.update_layout(title=f'Top-{k} Frequency', xaxis_title='Epoch', yaxis_title='Frequency')
            # for peak in peaks:
            #     plt.axvline(x=peak, color='r', linestyle='--', linewidth=1)
            st.session_state.plots.append(fig)
        st.session_state['init'] = True
    st.slider('k', 1, 20, key='k', value=1)
    st.plotly_chart(st.session_state['plots'][st.session_state['k']-1])
    
        

# # compute the mean attribution score over all inputs
# mean_attribution_scores = attribution_scores.mean(axis=1)

# # # sort in descending order
# sorted_mean_attribution_scores = np.sort(mean_attribution_scores, axis=-1)[...,::-1]
# improvement = mean_attribution_scores[1] - mean_attribution_scores[0]
# plt.figure()
# plt.hist(improvement, bins=30)
# plt.show()
# improvement += np.abs(improvement.min())
# improvement /= improvement.max()
# colors = np.zeros(len(mean_attribution_scores))
# colors = improvement

# # plot the mean attribution score, from top to bottom
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10), sharex=True, sharey=True)
# ax1.bar(np.arange(len(sorted_mean_attribution_scores[0])), sorted_mean_attribution_scores[0], color=cm.seismic(colors))#, width=2)
# ax2.bar(np.arange(len(sorted_mean_attribution_scores[1])), sorted_mean_attribution_scores[1], color=cm.RdBu(colors))#, width=2)
# ax1.set_title(f'Epoch 395')
# ax2.set_title(f'Epoch 400')
# ax2.set_xlabel('Neuron')
# plt.suptitle('Mean Attribution Score')
# plt.savefig("Single Layer ReLU/395_400_mean_attr_score.jpg")

# plot the cumulative mean attribution score, from top to bottom
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5), sharex=True, sharey=True)
# ax1.bar(np.arange(len(sorted_mean_attribution_scores[0]))[:20], np.cumsum(sorted_mean_attribution_scores[0])[:20]/np.sum(sorted_mean_attribution_scores[0]))
# ax2.bar(np.arange(len(sorted_mean_attribution_scores[1]))[:20], np.cumsum(sorted_mean_attribution_scores[1])[:20]/np.sum(sorted_mean_attribution_scores[1]))
# ax1.set_title(f'Epoch 395')
# ax1.grid(True)
# ax2.set_title(f'Epoch 400')
# ax2.grid(True)
# plt.suptitle('Cumulative Attribution Score')
# plt.savefig("Single Layer ReLU/395_400_cumulative_mean_attr_score.jpg")
