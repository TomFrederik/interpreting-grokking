from model import GrokkingTransformer
from torch import pca_lowrank
import matplotlib.pyplot as plt
import numpy as np
from utils import load_model
import matplotlib.animation as animation
import matplotlib.image as mgimg
import os
from tqdm import tqdm

model_name = 'Tied Embeddings'
_, ckpt_dir = load_model(model_name)
ckpts = [os.path.join(ckpt_dir, f"epoch={epoch}-step={epoch*10+9}.ckpt") for epoch in range(0,1050,5)]

os.makedirs(f"{model_name}/pca_embeddings", exist_ok=True)

for ckpt in tqdm(ckpts[-1:]):
    epoch = int(ckpt.split('/')[-1].split('-')[0].split('=')[-1])
    # if os.path.exists(f"{model_name}/pca_embeddings/epoch={epoch}.jpg"):
    #     continue
    model = GrokkingTransformer.load_from_checkpoint(ckpt)
    embedding = model.embedding.weight[:-1]

    mod = 32
    colors = np.arange(embedding.shape[0]) % mod

    # perform pca on embedding
    data = (embedding @ pca_lowrank(embedding, q=2)[-1]).detach().cpu().numpy()    
    
    max_num = data.shape[0]-1
    order = [0]
    cur = mod
    while cur != 0:
        order.append(cur)
        cur = (cur + mod) % 97
    
    plt.figure(figsize=(10,10))
    plt.title(f"Epoch {epoch}, mod {mod}")
    plt.scatter(data[:,0], data[:,1], alpha=0.5, c=colors, cmap='viridis')
    for i, txt in enumerate(map(lambda x: str(x), np.arange(data.shape[0]))):
        plt.annotate(txt, (data[i,0], data[i,1]))
        
        if i < data.shape[0]-1:
            plt.plot(data[order][i:i+2,0], data[order][i:i+2,1], alpha=0.3, c='grey')
        else:
            plt.plot([data[order][-1,0], data[order][0,0]], [data[order][i,1], data[order][0,1]], alpha=0.3, c='grey')
    plt.axis('off')
    plt.savefig(f"{model_name}/pca_embeddings/epoch={epoch}.jpg")
    plt.close()

# images = []
# fig = plt.figure()
# for i in range(0,1105,5):
#     image = mgimg.imread(f"{model_name}/pca_embeddings/epoch={i}.jpg")
#     images.append([plt.imshow(image)])
# plt.axis('off')
# my_anim = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=2000)
# my_anim.save(f'{model_name}/pca_embedding.gif', fps=2)