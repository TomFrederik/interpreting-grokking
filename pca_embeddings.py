from model import GrokkingTransformer
from sklearn.decomposition import PCA
import numpy as np
from utils import load_model
import os
from tqdm import tqdm

def pca_and_save(ckpt, epoch):
    model = GrokkingTransformer.load_from_checkpoint(ckpt)
    embedding = model.embedding.weight[:-1].detach().cpu().numpy()

    pca_model = PCA().fit(embedding)

    transformed_data = pca_model.transform(embedding)
    explained_variance_ratio = pca_model.explained_variance_ratio_

    np.savez_compressed(os.path.join(save_dir, f"embedding_pca_results_{epoch}.npz"), transformed_data=transformed_data, explained_variance_ratio=explained_variance_ratio)

model_name = 'Attention Only'
all_epochs = True
ckpt, ckpt_dir = load_model(model_name)

save_dir = f"{model_name}/pca_interface_data"
os.makedirs(save_dir, exist_ok=True)

if all_epochs:
    epochs = range(1090)
    for epoch in tqdm(epochs):
        save_path = f"{model_name}/pca_interface_data/embedding_pca_results_{epoch}.npz"
        ckpt = os.path.join(ckpt_dir, f"epoch={epoch}-step={epoch*10+9}.ckpt")
        pca_and_save(ckpt, epoch)
        
else:
    epoch = int(ckpt.split('/')[-1].split('-')[0].split('=')[-1])
    # if os.path.exists(f"{model_name}/pca_embeddings/epoch={epoch}.jpg"):
    #     continue
    pca_and_save(ckpt, epoch)
    