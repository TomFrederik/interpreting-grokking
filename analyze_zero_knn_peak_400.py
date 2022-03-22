import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr

attribution_scores = np.load(f"Single Layer ReLU/knn_zero_grads_395_400.npy")

inputs = [
    0,2,9,11,12,14,15,16,17,19,21,23,26,27,31,34,36,37,47,48,
    49,50,52,54,56,58,60,62,63,65,66,67,68,70,71,75,76,78,79,
    80,81,83,85,86,87,88,89,92,93,95,96
]

# compute correlation between attribution score rankings before and after peak
# ranking = np.argsort(attribution_scores, axis=-1)[...,::-1]
# corr1 = np.tril(spearmanr(ranking[0],axis=1)[0], k=-1)
# mean_corr1 = corr1[corr1 != 0].mean()
# corr2 = np.tril(spearmanr(ranking[1], axis=1)[0], k=-1)
# mean_corr2 = corr2[corr2 != 0].mean()

# diff_corr = corr2 - corr1
# diff_index_desc = np.argsort(diff_corr.flatten())[::-1]
# max_diff_idx = np.argmax(diff_corr)
# min_diff_idx = np.argmin(diff_corr)
# print(len(corr1.flatten()))
# print("Corr at 395   Corr at 400   Diff Corr   Input pair 1   Input pair 2") 
# for i in range(-10,0,1):
#     print(f"{corr1.flatten()[diff_index_desc[i]]:11.3f}", f"{corr2.flatten()[diff_index_desc[i]]:13.3f}", f"{diff_corr.flatten()[diff_index_desc[i]]:11.3f}", f"{inputs[diff_index_desc[i]//51]:14d}", f"{inputs[diff_index_desc[i]%51]:14d}")


def compute_mean_topk_overlap(rankings, k=10):
    """
    Compute the mean top-k overlap between all rankings.
    """
    num_rankings = rankings.shape[0]
    mean_topk_overlap = 0
    for i in range(num_rankings):
        for j in range(i+1, num_rankings):
            mean_topk_overlap += np.sum(np.isin(rankings[i][:k], rankings[j][:k]))
    return 2 * mean_topk_overlap / (num_rankings * (num_rankings-1) * k)

# overlap_1 = [compute_mean_topk_overlap(ranking[0], k) for k in range(1,len(ranking[0,0])+1)]
# overlap_2 = [compute_mean_topk_overlap(ranking[1], k) for k in range(1,len(ranking[1,0])+1)]

# plt.figure()
# plt.plot(np.arange(len(overlap_1)), overlap_1, label="395")
# plt.plot(np.arange(len(overlap_2)), overlap_2, label="400")
# plt.legend()
# plt.title('Mean Top-K Overlap')
# plt.xlabel('K')
# plt.ylabel('Overlap')
# plt.savefig('Single Layer ReLU/knn_zero_grads_395_400_mean_topk_overlap.jpg')

# compute the mean attribution score over all inputs
mean_attribution_scores = attribution_scores.mean(axis=1)

# # sort in descending order
sorted_mean_attribution_scores = np.sort(mean_attribution_scores, axis=-1)[...,::-1]
improvement = mean_attribution_scores[1] - mean_attribution_scores[0]
plt.figure()
plt.hist(improvement, bins=30)
plt.show()
improvement += np.abs(improvement.min())
improvement /= improvement.max()
colors = np.zeros(len(mean_attribution_scores))
colors = improvement

# plot the mean attribution score, from top to bottom
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10), sharex=True, sharey=True)
ax1.bar(np.arange(len(sorted_mean_attribution_scores[0])), sorted_mean_attribution_scores[0], color=cm.seismic(colors))#, width=2)
ax2.bar(np.arange(len(sorted_mean_attribution_scores[1])), sorted_mean_attribution_scores[1], color=cm.RdBu(colors))#, width=2)
ax1.set_title(f'Epoch 395')
ax2.set_title(f'Epoch 400')
ax2.set_xlabel('Neuron')
plt.suptitle('Mean Attribution Score')
plt.savefig("Single Layer ReLU/395_400_mean_attr_score.jpg")

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