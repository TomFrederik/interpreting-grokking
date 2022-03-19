from collections import Counter
import itertools as it
import os

import numpy as np
import einops 
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from model import GrokkingTransformer
from datasets import get_dataset
from utils import load_model

@torch.no_grad()
def compute_logit_matrix(model, dataset):
    """Computes logits for each output token for each sample in the dataset

    :param model: model that's investigated
    :type model: GrokkingTransformer
    """
    return einops.rearrange(model(torch.from_numpy(dataset.data[:,:-1]).to(model.device))[:,-1], 'a b -> (a b)').cpu()

def compute_all_matrices(ckpt_dir, dataset, device):
    # get list of ckpt files in ckpt_dir
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    files = sorted(files, key=lambda x: int(x[6:].split('-')[0]))
    total_logit_matrix = []
    for i, f in enumerate(tqdm(files)):
        ckpt = os.path.join(ckpt_dir, f)
        model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
        model.eval()
        logit_matrix = compute_logit_matrix(model, dataset)
        total_logit_matrix.append(logit_matrix)
    return torch.stack(total_logit_matrix, dim=0)




model_name = "Single Layer ReLU"
ckpt, ckpt_dir = load_model(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_name = 'minus'
num_elements = 97
data_dir = './data'
force_data = False
dataset = get_dataset(descr=data_name, num_elements=num_elements, data_dir=data_dir, force_data=force_data)

np.random.seed(42)

max_dim = 10
max_epoch = 1200
try:
    pca_matrix = np.load(os.path.join(f"{model_name}", 'pca_matrix.npy'))
except:
    try:
        total_logit_matrix = torch.load(f'{model_name}/total_logit_matrix.pt')
    except:
        total_logit_matrix = compute_all_matrices(ckpt_dir=ckpt_dir, dataset=dataset, device=device)
        torch.save(total_logit_matrix, f'{model_name}/total_logit_matrix.pt')
    total_logit_matrix = total_logit_matrix[:max_epoch].numpy()
    pca = PCA(n_components=max_dim)
    centered_data = total_logit_matrix - np.mean(total_logit_matrix, axis=0)
    pca = pca.fit(centered_data)
    transformed_data = pca.transform(centered_data)
    np.savez_compressed(os.path.join(f"{model_name}", 'functionspace_pca_results.npz'), transformed_data=transformed_data, explained_variance_ratio=pca.explained_variance_ratio_)


# dim = 3
# # pca = PCA(n_components=dim).fit(total_logit_matrix - total_logit_matrix.mean(axis=0))
# # u = pca.transform(total_logit_matrix - total_logit_matrix.mean(axis=0))[:,:dim]
# (u, s, v) = torch.pca_lowrank(torch.from_numpy(total_logit_matrix[:10]), q=1200, niter=2)
# # print(f"{s.shape = }")
# # print(s)
# # print(s.sum())
# u = u.numpy()[:,:dim] # this is the projected matrix
# colors = np.linspace(0, 1, len(u))

# cluster1 = np.arange(394,405)

# x_range = np.arange(97)
# y_range = np.arange(97)
# xy_range = it.product(x_range, y_range)
# z = np.array(list(map(lambda xy: (xy[0] - xy[1]) % 97, xy_range)))


# if dim == 3:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(u[:,0], u[:,1], u[:,2], c=cm.viridis(colors))
#     ax.scatter(u[cluster1,0], u[cluster1,1], u[cluster1, 2], c='r', s=3)
#     for t in cluster1:
#         ax.text(u[t,0], u[t,1], u[t,2], str(t), color='r')
#     plt.show()
# elif dim == 2:
#     plt.scatter(u[:,0], u[:,1], c=cm.viridis(colors), s=3)
#     step_size = 20

#     plt.scatter(u[cluster1,0], u[cluster1,1], c='r', s=3)
#     for t in cluster1:
#         plt.annotate(str(t), (u[t,0], u[t,1]), c='r')

#     cb = plt.colorbar(ticks=np.arange(0,len(u),100)/len(u), orientation='horizontal', label='Epoch')
#     cb.ax.set_xticklabels(np.arange(0,len(u),100))
#     plt.show()
#     # plt.savefig(f'{model_name}/functionspace_pca.jpg')
