import numpy as np
import matplotlib.pyplot as plt
import torch

import circuits_util as cutil
from utils import load_model
from model import GrokkingTransformer, MultiheadAttention

ckpt, _ = load_model()

model = GrokkingTransformer.load_from_checkpoint(ckpt) 

mah = model.transformer[0].self_attn
assert isinstance(mah, MultiheadAttention)

for i in range(4):
    ov = cutil.get_ov_circuit(model, layer=0, head=i)[:-2,:-2]
    print(f"{ov.shape = }")

    plt.figure()
    plt.imshow(ov.detach().cpu().numpy(), origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0,97,8))
    plt.yticks(np.arange(0,97,8))
    plt.show()
    plt.close()