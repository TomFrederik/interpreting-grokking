import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr

attribution_scores = np.load(f"Single Layer ReLU/knn_zero_grads_395_400.npy")

# compute correlation between attribution score rankings before and after peak
ranking = np.argsort(attribution_scores, axis=-1)[...,::-1]
corr1 = np.tril(spearmanr(ranking[0],axis=1)[0], k=-1)
mean_corr1 = corr1[corr1 != 0].mean()
corr2 = np.tril(spearmanr(ranking[1], axis=1)[0], k=-1)
mean_corr2 = corr2[corr2 != 0].mean()
print(mean_corr1)
print(mean_corr2)

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

overlap_1 = [compute_mean_topk_overlap(ranking[0], k) for k in range(1,len(ranking[0,0])+1)]
overlap_2 = [compute_mean_topk_overlap(ranking[1], k) for k in range(1,len(ranking[1,0])+1)]

# plt.figure()
# plt.plot(np.arange(len(overlap_1)), overlap_1, label="395")
# plt.plot(np.arange(len(overlap_2)), overlap_2, label="400")
# plt.legend()
# plt.title('Mean Top-K Overlap')
# plt.xlabel('K')
# plt.ylabel('Overlap')
# plt.savefig('Single Layer ReLU/knn_zero_grads_395_400_mean_topk_overlap.jpg')

# compute the mean attribution score over all inputs
# mean_attribution_scores = attribution_scores.mean(axis=1)

# # sort in descending order
# sorted_mean_attribution_scores = np.sort(mean_attribution_scores, axis=-1)[...,::-1]
# colors = np.zeros(len(mean_attribution_scores[0]))
# top_movers = np.argsort(mean_attribution_scores[1] - mean_attribution_scores[0], axis=-1)[:10]
# colors[top_movers] = 1
# sort_idcs = np.argsort(mean_attribution_scores, axis=-1)[...,::-1]

# plot the mean attribution score, from top to bottom
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5), sharex=True, sharey=True)
# ax1.bar(np.arange(len(sorted_mean_attribution_scores[0])), sorted_mean_attribution_scores[0], color=cm.viridis(colors[sort_idcs[0]]))
# ax2.bar(np.arange(len(sorted_mean_attribution_scores[1])), sorted_mean_attribution_scores[1], color=cm.viridis(colors[sort_idcs[1]]))
# ax1.set_title(f'Epoch 395')
# ax2.set_title(f'Epoch 400')
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